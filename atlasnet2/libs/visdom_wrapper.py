import visdom
import torch


class VisdomWrapper:
    def __init__(self, server: str = "http://localhost", port: int = 8097, env: str = "main"):
        self._vis = visdom.Visdom(server=server, port=port, env=env)
        self._is_alive = self._vis.check_connection()

        self._graphs = dict()

    def append_point_to_curve(self, graph: str, curve: str, x: float, y: float):
        pass

    def show_graph(self, graph: str):
        pass

    def show_point_cloud(self, name: str, point_cloud: torch.Tensor):
        if self._is_alive:
            self._vis.scatter(X=point_cloud[0].data.cpu(),
                              win=name,
                              opts=dict(
                                  title=name,
                                  markersize=2,
                                  webgl=True
                              ))
