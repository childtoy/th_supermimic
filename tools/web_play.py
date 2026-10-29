#!/usr/bin/env python3
"""Local live policy controller. GPU simulation runs independently of HTTP rendering."""
import argparse
import asyncio
from collections import deque
import io
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class LiveRenderer:
    """Own GL on a separate thread; retain only the most recent pose."""
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()
        self.pose = None
        self.ms = 0.0
        self.hz = 0.0
        self.error = None
        self.renderer_name = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def submit(self, policy_qpos, reference_qpos):
        # Never wait for a render to complete and never build a frame backlog.
        with self.lock:
            self.pose = (policy_qpos, reference_qpos)

    def run(self):
        renderer = None
        try:
            import mujoco
            from PIL import Image
            from th_supermimic.tasks.motion_tracking.humanoid.supermimic.render import load_comparison_model
            a = self.engine.args
            model = load_comparison_model(a.assets_dir / 'scene.xml')
            model.vis.global_.offwidth = max(a.render_width, model.vis.global_.offwidth)
            model.vis.global_.offheight = max(a.render_height, model.vis.global_.offheight)
            model.vis.headlight.ambient[:] = .45
            model.vis.headlight.diffuse[:] = .55
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, width=a.render_width, height=a.render_height)
            try:
                from OpenGL.GL import glGetString, GL_RENDERER
                name = glGetString(GL_RENDERER)
                self.renderer_name = name.decode() if name else None
            except Exception:
                self.renderer_name = None
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 3.6, 135, -12
            samples, timestamps = deque(maxlen=40), deque(maxlen=40)
            while not self.engine.stop.is_set():
                start = time.perf_counter()
                with self.lock:
                    pose = self.pose
                    self.pose = None
                if pose is None:
                    self.engine.stop.wait(.01)
                    continue
                data.qpos[:model.nq//2], data.qpos[model.nq//2:] = pose
                mujoco.mj_kinematics(model, data)
                camera.lookat[:] = [data.qpos[0], data.qpos[1], .75]
                renderer.update_scene(data, camera=camera)
                buffer = io.BytesIO()
                Image.fromarray(renderer.render()).save(buffer, format='JPEG', quality=75)
                end = time.perf_counter()
                samples.append((end-start)*1000)
                timestamps.append(end)
                with self.engine.lock:
                    self.engine.jpeg = buffer.getvalue()
                self.ms = sum(samples)/len(samples)
                self.hz = (len(timestamps)-1)/(timestamps[-1]-timestamps[0]) if len(timestamps)>1 else 0.0
                self.engine.stop.wait(max(0,1/a.render_fps-(end-start)))
        except Exception as exc:
            traceback.print_exc()
            self.error = str(exc)
        finally:
            if renderer is not None:
                renderer.close()


class LiveEngine:
    def __init__(self, args):
        self.args = args
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.snapshot = dict(status='loading', actions=[], events=[])
        self.jpeg = b''
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        env = None
        display = None
        try:
            import numpy as np
            import torch
            from th_supermimic.data.command_bank import CommandBank, reference_timing
            from th_supermimic.sim.env import TrackingEnv
            from th_supermimic.rl.algos.supermimic.geometry import unflatten_state
            from th_supermimic.tasks.motion_tracking.humanoid.supermimic.evaluate import load_policy
            from th_supermimic.tasks.motion_tracking.humanoid.supermimic.render import flattened_qpos, reference_qpos
            a = self.args
            manifest = json.loads((a.motion_dir / 'manifest.json').read_text())
            records = manifest['clips']
            files = [str((a.motion_dir / (r['name'] + '.csv')).resolve()) for r in records]
            saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
            torch.manual_seed(42)
            env = TrackingEnv(a.assets_dir, files, num_envs=1, backend=a.backend, eval_mode=True, randomization=False)
            policy = load_policy(env, saved)
            bank = CommandBank(env, records)
            lib, mdp = env.motion, env.supermimic_mdp
            display = LiveRenderer(self)
            display.thread.start()
            actions = sorted(bank.primitives)
            history = None
            paused, failed, pending = True, False, None
            at_end = False
            active = actions[0]
            target_start = 0
            transition_start = None
            sim_steps = 0
            events = deque(maxlen=12)
            times = deque(maxlen=100)
            was_advanced = False
            latencies = deque(maxlen=100)
            rates = deque(maxlen=100)
            last_render = 0.0
            elapsed_start = time.perf_counter()

            def restart():
                nonlocal history, active, target_start, sim_steps, pending, failed, elapsed_start, paused, at_end, transition_start
                paused, at_end = True, False
                transition_start = None
                bank.restore()
                active = actions[0]
                target_start = int(lib.clip_start[bank.primitives[active]])
                env.reset_to_frames(torch.tensor([target_start], device=env.device))
                history, pending, failed, sim_steps = None, None, False, 0
                events.clear(); times.clear(); latencies.clear(); rates.clear()
                elapsed_start = time.perf_counter()

            restart()
            with torch.inference_mode():
                while not self.stop.is_set():
                    tick_start = time.perf_counter()
                    advanced = False
                    latest = None
                    while True:
                        try:
                            command = self.commands.get_nowait()
                        except queue.Empty:
                            break
                        if command['action'] == 'restart':
                            restart()
                            latest = None
                        elif command['action'] == 'pause':
                            paused = not paused
                        elif command['action'] in actions:
                            latest = command
                    if latest is not None:
                        pending = latest
                        if not failed:
                            paused = False
                    f = int(env.f[0])
                    in_transition = f < target_start
                    if pending is not None and not paused and not failed and not in_transition:
                        request = pending
                        pending = None
                        if request['action'] != active:
                            switch_start = time.perf_counter()
                            anchor, target_start, edge = bank.switch(active, request['action'], (f-target_start)/50, f)
                            transition_start = anchor
                            env.f[0] = anchor
                            env._publish()
                            # Explicit synchronization makes measured selection time include GPU work.
                            if env.device.type == 'cuda':
                                torch.cuda.synchronize(env.device)
                            compute_ms = (time.perf_counter()-switch_start)*1000
                            latency_ms = (time.perf_counter()-request['received'])*1000
                            latencies.append(latency_ms)
                            events.appendleft(dict(source=active, target=request['action'], sim_s=sim_steps/50,
                                                   compute_ms=compute_ms, receive_to_apply_ms=latency_ms,
                                                   bridge_s=edge['bridge']/30, phase_error_s=edge['phase_error_s']))
                            active = request['action']
                            at_end = False
                            f = anchor
                    if at_end and not pending:
                        paused = True
                    if not paused and not failed:
                        current = env.info()
                        obs = mdp.obs(unflatten_state(current['s']), current['a_prev'], current['xi'], current['f'])
                        history = obs[:, None].repeat(1, saved['args']['hist'], 1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                        action = policy(history, mdp.goal(current['f'], saved['args']['goal_frames']))
                        # Drive physics without auto-reset. End of reference holds its final pose.
                        target = env.default_q + env.alpha * action
                        env.sim.step(target)
                        end = int(lib.clip_end[lib.clip_id[f]])
                        env.f[0] = min(f+1, end)
                        env.a_prev = action.clone()
                        env.ep_step += 1
                        failed = bool(mdp.termination(env.read(), env.f)[0])
                        env._publish()
                        advanced = True
                        sim_steps += 1
                        if int(env.f[0]) >= end:
                            at_end, paused = True, True
                    compute_end = time.perf_counter()
                    if advanced and not was_advanced:
                        rates.clear()
                    if advanced:
                        rates.append((compute_end, sim_steps))
                    was_advanced = advanced
                    if advanced:
                        times.append((compute_end-tick_start)*1000)
                    now = time.perf_counter()
                    if now-last_render >= 1/a.render_fps:
                        start_render = now
                        f = int(env.f[0])
                        policy_pose = flattened_qpos(env.info()['s'])[0].cpu().numpy().copy()
                        reference_pose = reference_qpos(lib, env.f)[0].cpu().numpy().copy()
                        display.submit(policy_pose, reference_pose)
                        last_render = now
                        timing = reference_timing(f, target_start, int(lib.clip_end[lib.clip_id[f]]), transition_start)
                        state = dict(status='failed' if failed else 'waiting' if at_end else 'paused' if paused else 'running', actions=actions,
                                     active=active, phase='transition' if f<target_start else 'playing',
                                     pending=pending['action'] if pending else None, sim_s=sim_steps/50,
                                     **timing,
                                     primitive_durations={name:float((lib.clip_end[i]-lib.clip_start[i]).item())/50 for name,i in bank.primitives.items()},
                                     end_behavior='pause_simulation',
                                     compute_ms=sum(times)/max(1,len(times)), render_ms=display.ms,
                                     render_hz=display.hz, render_error=display.error,
                                     renderer_name=display.renderer_name,
                                     renderer_decoupled=True,
                                     simulated_steps_per_wall_second=0.0 if paused or failed or len(rates)<2 else (rates[-1][1]-rates[0][1])/max(.001,rates[-1][0]-rates[0][0]),
                                     events=list(events), backend=a.backend,
                                     frame_at=time.time())
                        with self.lock:
                            self.snapshot = state
                    self.stop.wait(max(0, .02-(time.perf_counter()-tick_start)))
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.snapshot = dict(status='error', error=str(exc), actions=[], events=[])
        finally:
            if display is not None:
                self.stop.set()
                display.thread.join(timeout=10)
            if env is not None:
                env.close()


async def serve(a):
    from aiohttp import web
    engine = LiveEngine(a)
    engine.thread.start()
    async def index(request):
        return web.FileResponse(ROOT/'web/index.html')
    async def state(request):
        with engine.lock:
            data = dict(engine.snapshot)
        return web.json_response(data, headers={'Cache-Control':'no-store'})
    async def frame(request):
        with engine.lock:
            image = engine.jpeg
        return web.Response(body=image, content_type='image/jpeg', headers={'Cache-Control':'no-store'})
    async def command(request):
        data = await request.json()
        action = data.get('action')
        with engine.lock:
            allowed = engine.snapshot.get('actions', []) + ['pause','restart']
        if action not in allowed:
            raise web.HTTPBadRequest(text='Unknown action')
        if engine.commands.qsize() >= 100:
            raise web.HTTPTooManyRequests()
        received = time.perf_counter()
        engine.commands.put(dict(action=action, received=received))
        return web.json_response(dict(accepted=True, action=action))
    app = web.Application(client_max_size=1024)
    app.add_routes([web.get('/',index),web.get('/api/state',state),web.get('/api/frame',frame),web.post('/api/command',command)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner,a.host,a.port)
    await site.start()
    print(f'Live controller: http://{a.host}:{a.port}',flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        engine.stop.set()
        await asyncio.to_thread(engine.thread.join, 10)
        await runner.cleanup()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot',choices=['k1','g1'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--motion-dir',type=Path,required=True)
    p.add_argument('--assets-dir',type=Path,required=True)
    p.add_argument('--backend',choices=['mjwarp','mujoco'],default='mjwarp')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=6006)
    p.add_argument('--render-fps',type=float,default=10)
    p.add_argument('--render-width',type=int,default=640)
    p.add_argument('--render-height',type=int,default=384)
    a=p.parse_args()
    if not 160<=a.render_width<=1920 or not 120<=a.render_height<=1080:
        p.error('Render dimensions out of range')
    if not 1<=a.render_fps<=50:
        p.error('--render-fps must be 1..50')
    os.environ['TH_SUPERMIMIC_ROBOT']=a.robot
    os.environ['TH_SUPERMIMIC_ASSETS']=str(a.assets_dir.resolve())
    os.environ.setdefault('MUJOCO_GL','egl')
    asyncio.run(serve(a))


if __name__=='__main__':
    main()
