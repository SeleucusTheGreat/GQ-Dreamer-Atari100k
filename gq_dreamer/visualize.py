"""Dream visualisation: decoded dream frames with actions, predicted rewards, p(continue) and advantages. The GPU part
(decoding, heads) runs here; the drawing is dream_render's (matplotlib only, so it can run in a worker process)."""
import os

import numpy as np
import torch

from .dream_render import render_dreams_pdf
from .model import PixelDecoder


@torch.no_grad()
def dream_data(agent, states, rewards, actions, advantages=None, label=None, max_advantage=None, action_names=None):
    """One dream -> numpy data for dream_render: decoded frames, p(continue), rewards, actions, advantages."""
    st = states.to(agent.device).float()
    zf = st[:, agent.tssm_dim:]
    with agent._amp():
        frames = PixelDecoder.to_image(agent.tokenizer.decode(zf)).cpu()
    alive = agent.alive_prob(st).cpu()
    to_np = lambda x: x.detach().float().cpu().numpy()                 # noqa: E731
    return {"frames": np.clip(frames.float().numpy(), 0, 1), "rewards": to_np(rewards),
            "actions": torch.as_tensor(actions).argmax(-1).cpu().numpy(), "alive": alive.numpy(),
            "adv": to_np(advantages) if advantages is not None else None, "label": label,
            "max_advantage": None if max_advantage is None else float(max_advantage),
            "action_names": tuple(action_names) if action_names is not None else None,
            "game": getattr(agent.cfg, "game", "")}


def save_dreams_pdf(agent, dreams, directory, keep=10, executor=None, action_names=None):
    """Writes packed dreams (metric, states, rewards, values, actions, advantages, normalised advantages, label) to
    <directory>/dream_g<updates>.pdf, keeping the newest `keep` PDFs -> path. executor: the drawing is submitted to it
    (a worker process); the future is stored in save_dreams_pdf.pending."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"dream_g{agent.total_num_updates}.pdf")
    data = [dream_data(agent, states, rewards, actions, advantages=norm_adv, label=label, max_advantage=metric,
                       action_names=action_names)
            for metric, states, rewards, _values, actions, _adv, norm_adv, label in dreams]
    if executor is None:
        return render_dreams_pdf(path, data, directory, keep)
    save_dreams_pdf.pending = executor.submit(render_dreams_pdf, path, data, directory, keep)
    return path


save_dreams_pdf.pending = None
