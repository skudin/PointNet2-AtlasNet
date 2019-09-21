import visdom
import torch
import numpy as np


class VisdomWrapper:
    def __init__(self, server: str = "http://localhost", port: int = 8097, env: str = "main"):
        self._vis = visdom.Visdom(server=server, port=port, env=env)
        self._is_alive = self._vis.check_connection()

        self._graphs = dict()

    def append_point_to_curve(self, graph: str, curve: str, x: float, y: float):
        if graph in self._graphs:
            if curve in self._graphs[graph]:
                self._graphs[graph][curve]["x"].append(x)
                self._graphs[graph][curve]["y"].append(y)
            else:
                self._graphs[graph][curve] = {"x": [x], "y": [y]}
        else:
            self._graphs[graph] = {curve: {"x": [x], "y": [y]}}

    def show_graph(self, graph: str):
        if self._is_alive:
            x_values = np.column_stack([np.array(self._graphs[graph][key]["x"]) for key in self._graphs[graph]])
            y_values = np.column_stack([np.array(self._graphs[graph][key]["y"]) for key in self._graphs[graph]])
            legend = [key for key in self._graphs[graph]]

            self._vis.line(X=x_values, Y=y_values, win=graph, opts=dict(title=graph, legend=legend, markersize=2))

    def show_point_cloud(self, name: str, point_cloud: torch.Tensor):
        if self._is_alive:
            self._vis.scatter(X=point_cloud[0].data.cpu(),
                              win=name,
                              opts=dict(
                                  title=name,
                                  markersize=2,
                                  webgl=True
                              ))

    def show_image(self, name: str, image: torch.Tensor):
        if self._is_alive:
            self._vis.image(image[0].data.cpu(), win=name, opts=dict(title=name))
