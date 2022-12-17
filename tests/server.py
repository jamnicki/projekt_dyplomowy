from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime as dt


class RequestHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

    def do_GET(self):
        self.do_HEAD()


def main():
    httpd = HTTPServer(('localhost', 8000), RequestHandler)
    for _ in range(5):
        httpd.handle_request()
        print(f"{dt.now()}: Started next iteration.")


if __name__ == "__main__":
    main()
