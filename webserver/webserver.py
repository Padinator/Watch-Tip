from http.server import HTTPServer, BaseHTTPRequestHandler


class Serv(BaseHTTPRequestHandler):

    def __init__(self, request, client_address, server):
        super().__init__(request, client_address, server)

    def do_GET(self) -> None:
        """
        Handle HTTP-GET requests.
        """

        if self.path == '/':  # Default route is default page
            self.path = '/index.html'

        try:
            file_to_open = open(self.path[1:]).read()
            self.send_response(200)
        except Exception:
            file_to_open = "Route not found"
            self.send_response(404)

        self.end_headers()
        self.wfile.write(bytes(file_to_open, 'utf-8'))


try:
    print("Start server")
    httpd = HTTPServer(('localhost', 8080), Serv)
    httpd.serve_forever()
except KeyboardInterrupt:
    print("Terminate server")
