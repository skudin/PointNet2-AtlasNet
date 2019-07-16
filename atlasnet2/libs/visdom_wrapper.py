import visdom


class VisdomWrapper:
    def __init__(self, server: str = "http://localhost", port: int = 8097, env: str = "main"):
        self._vis = visdom.Visdom(server=server, port=port, env=env)
        self._is_alive = self._vis.check_connection()
