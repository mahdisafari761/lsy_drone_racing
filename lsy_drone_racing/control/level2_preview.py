"""Correct an observed collision risk using an independent nominal flight model.

The private model is initialized from public configuration and observations only.
It is never given a race environment, hidden object poses or randomized parameters.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow import Sim
from crazyflow.sim.functional import attitude_control
from jax.scipy.spatial.transform import Rotation

from lsy_drone_racing.control._reference_policy_jax import actor, reference_features
from lsy_drone_racing.control.level2_lead import Level2Lead


@lru_cache(maxsize=8)
def build_preview(
    drone: str, dynamics: str, sim_freq: int, attitude_freq: int, freq: int,
    horizon: int, margin: float, strength: float, bounds: tuple,
) -> tuple:
    """Compile public-observation policy rollouts in a separate nominal simulator."""
    samples = np.zeros((19,4),np.float32)
    angles = np.arange(8)*np.pi/4
    samples[1:9,:2] = strength*np.c_[np.cos(angles),np.sin(angles)]
    samples[9:17,:2] = 2*strength*np.c_[np.cos(angles),np.sin(angles)]
    samples[17:,3] = [-strength,strength]
    offsets, count = jnp.asarray(samples), len(samples)
    sim = Sim(n_worlds=count,drone=drone,dynamics=dynamics,control="attitude",
              freq=sim_freq,attitude_freq=attitude_freq)
    step_model = sim.build_step_fn()
    frame_centers = jnp.array([[0,0,.28],[0,0,-.28],[0,-.28,0],[0,.28,0]])
    frame_half = jnp.array([[.01,.36,.08],[.01,.36,.08],[.01,.08,.36],[.01,.08,.36]])
    drone_half = jnp.array([.07,.07,.02])
    low, high = map(jnp.asarray,bounds)

    @jax.jit
    def predict(data: Any, public: dict, previous: jax.Array, params: dict,
                gains: jax.Array, mass: jax.Array, thrust_bounds: jax.Array) -> tuple:
        observations = jax.tree.map(lambda x:jnp.broadcast_to(x,(count,)+x.shape),public)
        data = data.replace(states=data.states.replace(**{
            key:observations[key][:,None] for key in ["pos","vel","quat","ang_vel"]}),
            params=data.params.replace(mass=jnp.full_like(data.params.mass,mass)))
        previous = jnp.broadcast_to(previous,(count,4))
        gate_rot = Rotation.from_quat(public["gates_quat"]).as_matrix()
        batch = jnp.arange(count)

        def advance(carry: tuple, tick: jax.Array) -> tuple:
            data, obs, previous, minimum, initial_data, initial_raw, initial_action = carry
            phase = jnp.minimum(obs["n_gates_passed"],4)
            settings = gains[phase]
            encoded = reference_features(obs,previous)
            encoded = encoded.at[:,:3].multiply(settings[:,0,None])
            encoded = encoded.at[:,15:18].multiply(settings[:,3,None])
            encoded = encoded.at[:,25:28].multiply(settings[:,3,None])
            raw = actor(params,encoded)
            raw = jnp.clip(raw+settings[:,1,None]*(raw-previous)
                           +offsets*jnp.maximum(0.,1.-tick/10.),-1.,1.)
            thrust = thrust_bounds[0]+.5*(raw[:,3]+1)*(thrust_bounds[1]-thrust_bounds[0])
            action = jnp.concatenate([raw[:,:2]*params["tilt_max"],raw[:,2:3]*.5,
                jnp.clip(thrust+settings[:,2],*thrust_bounds)[:,None]],axis=-1)
            before = data.states.pos[:,0]
            data = step_model(attitude_control(data,action[:,None]),sim_freq//freq)
            initial_data = jax.lax.cond(tick==0,lambda _:data,lambda _:initial_data,None)
            initial_raw = jnp.where(tick==0,raw,initial_raw)
            initial_action = jnp.where(tick==0,action,initial_action)
            position = data.states.pos[:,0]
            local = jnp.einsum("gji,ngj->ngi",gate_rot,
                               position[:,None]-public["gates_pos"][None])
            body = Rotation.from_quat(data.states.quat[:,0]).as_matrix()
            relative = jnp.einsum("gji,njk->ngik",gate_rot,body)
            projected = jnp.einsum("ngij,j->ngi",jnp.abs(relative),drone_half)
            gap = jnp.abs(local[:,:,None]-frame_centers)-frame_half-projected[:,:,None]
            distance = jnp.linalg.norm(jnp.maximum(gap,0.),axis=-1)+jnp.minimum(
                jnp.max(gap,axis=-1),0.)
            # Exact observed objects are protected; unknown map entries cannot yet
            # be treated as exact collision locations.
            frame = jnp.min(jnp.where(public["gates_visited"][None,:,None],distance,10.),axis=(1,2))
            world_half = jnp.einsum("nij,j->ni",jnp.abs(body),drone_half)
            gap = jnp.abs(position[:,None,:2]-public["obstacles_pos"][None,:,:2])-world_half[:,None,:2]
            poles = jnp.linalg.norm(jnp.maximum(gap,0.),axis=-1)+jnp.minimum(jnp.max(gap,axis=-1),0.)-.015
            poles = jnp.where(public["obstacles_visited"][None]
                &(position[:,None,2]-world_half[:,None,2]<public["obstacles_pos"][None,:,2]),poles,10.)
            boundary = jnp.minimum(jnp.min(position-low,axis=-1),jnp.min(high-position,axis=-1))
            clearance = jnp.minimum(jnp.minimum(frame,jnp.min(poles,axis=-1)),boundary+margin)
            clearance = jnp.where(obs["n_gates_passed"]>=5,10.,clearance)
            minimum = jnp.minimum(minimum,clearance)
            gid = obs["gate_sequence"][batch,phase]
            direction = obs["gate_sequence_direction"][batch,phase]
            start_local = jnp.einsum("nji,nj->ni",gate_rot[gid],before-public["gates_pos"][gid])
            end_local = local[batch,gid]
            fraction = jnp.clip(-start_local[:,0]/(end_local[:,0]-start_local[:,0]+1e-8),0.,1.)
            crossing = start_local[:,1:]+fraction[:,None]*(end_local[:,1:]-start_local[:,1:])
            crossed = (start_local[:,0]*direction<=0)&(end_local[:,0]*direction>0)
            crossed &= jnp.all(jnp.abs(crossing)<.2,axis=-1)
            next_obs = dict(obs)
            next_obs.update({key:getattr(data.states,key)[:,0] for key in ["pos","vel","quat","ang_vel"]})
            next_obs["n_gates_passed"] = jnp.minimum(obs["n_gates_passed"]+crossed,5)
            return (data,next_obs,raw,minimum,initial_data,initial_raw,initial_action),None

        initial = (data,observations,previous,jnp.full(count,10.),data,
                   jnp.zeros((count,4)),jnp.zeros((count,4)))
        final,_ = jax.lax.scan(advance,initial,jnp.arange(horizon))
        _,end,_,clearance,first_data,first_raw,first_action = final
        passed = end["n_gates_passed"]
        gid = public["gate_sequence"][jnp.minimum(passed,4)]
        distance = jnp.linalg.norm(end["pos"]-public["gates_pos"][gid],axis=-1)
        risk = jnp.maximum(margin-clearance,0.)
        cost = 1e5*risk**2+jnp.sum(offsets**2,axis=-1)+.005*distance-.02*passed
        selected = jnp.where(clearance[0]>=margin,0,jnp.argmin(cost))
        def choose(x: jax.Array) -> jax.Array:
            return jnp.broadcast_to(x[selected][None],x.shape) if x.ndim and x.shape[0]==count else x
        return jax.tree.map(choose,first_data),first_raw[selected],first_action[selected],selected,clearance[0]

    return sim.data,predict


class Level2Preview(Level2Lead):
    """Keep the original action unless a private short rollout predicts collision."""

    HORIZON = 15
    MARGIN = .01
    STRENGTH = .06

    def __init__(self, obs: dict, info: dict, config: dict):
        """Initialize nominal model and compile outside the ordinary control loop."""
        super().__init__(obs,info,config)
        with np.load(Path(__file__).parent/"policies"/self.POLICY_FILE,allow_pickle=False) as archive:
            self.parameters={key:jnp.asarray(archive[key]) for key in archive.files}
        limits=config.env.track.safety_limits
        bounds=(tuple(limits.pos_limit_low),tuple(limits.pos_limit_high))
        self.model,self.predict=build_preview(config.sim.drone,config.sim.dynamics,
            config.sim.freq,config.sim.attitude_freq,config.env.freq,self.HORIZON,
            self.MARGIN,self.STRENGTH,bounds)
        self.mass=.0434
        self.last_velocity=None
        self.last_command=None
        self.frequency=config.env.freq
        self.tick=0
        self.interventions=0
        self.predict(self.model,{k:jnp.asarray(v) for k,v in obs.items()},jnp.asarray(self.previous),
            self.parameters,jnp.asarray(self.gains),jnp.asarray(self.mass),
            jnp.asarray([self.thrust_low,self.thrust_high]))[1].block_until_ready()

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Estimate mass from observed impulse and select a nearby safe command."""
        if self.last_velocity is not None and self.tick>8 and obs["pos"][2]>.12:
            az=(obs["vel"][2]-self.last_velocity[2])*self.frequency+9.81
            if az>3.:
                qx,qy=np.asarray(obs["quat"])[:2]
                estimate=self.last_command[3]*(1-2*(qx*qx+qy*qy))/az
                self.mass=.95*self.mass+.05*np.clip(estimate,.0384,.0484)
        self.model,raw,action,index,_=self.predict(self.model,
            {k:jnp.asarray(v) for k,v in obs.items()},jnp.asarray(self.previous),
            self.parameters,jnp.asarray(self.gains),jnp.asarray(self.mass),
            jnp.asarray([self.thrust_low,self.thrust_high]))
        self.previous=np.asarray(raw)
        self.last_velocity=np.asarray(obs["vel"]).copy()
        self.last_command=np.asarray(action)
        self.interventions+=int(index)>0
        self.tick+=1
        return self.last_command.astype(np.float32)
