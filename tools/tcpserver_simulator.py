#!/usr/bin/env python3
"""
Openterface Server Simulator
Simulates the TCP server in the server module, handling lastImage commands
Supports updating image paths via HTTP interface
"""

import socket
import threading
import os
import sys
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse

# Global variable to store current image path
current_image_path = r"images/4a.jpg"
image_path_lock = threading.Lock()


def read_image_file(file_path):
    """
    Read the specified image file
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        bytes: Binary data of the image file
    """
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading image file: {e}")
        return None

def handle_client(client_socket, address):
    """
    Handle client connections
    
    Args:
        client_socket: Client socket connection
        address: Client address
    """
    print(f"✅ Client {address} connected")
    
    try:
        # Receive command from client
        command = client_socket.recv(1024).decode('utf-8', errors='ignore').strip()
        print(f"📥 Received command: {command}")
        
        # Only handle lastimage command
        if command.lower() == "lastimage":
            # Use global image path
            with image_path_lock:
                image_path = current_image_path
            
            # Check if image file exists
            if not os.path.exists(image_path):
                error_msg = f"Image file does not exist: {image_path}"
                client_socket.send(f"ERROR: {error_msg}\n".encode('utf-8'))
                print(f"❌ {error_msg}")
            else:
                # Read image file
                image_data = read_image_file(image_path)
                if image_data is None:
                    error_msg = "Failed to read image file"
                    client_socket.send(f"ERROR: {error_msg}\n".encode('utf-8'))
                    print(f"❌ {error_msg}")
                else:
                    # Send image data to client - using correct binary format
                    image_size = len(image_data)
                    # First send header information, ensure it ends with newline
                    header = f"IMAGE:{image_size}\n"
                    client_socket.send(header.encode('utf-8'))
                    # Then send image data - ensure it's complete binary data
                    client_socket.send(image_data)
                    print(f"📤 Image sent, size: {image_size} bytes")
                    print(f"✅ Image transfer completed")
        else:
            # Unsupported command
            error_msg = f"Unsupported command: {command}"
            client_socket.send(f"ERROR: {error_msg}\n".encode('utf-8'))
            print(f"❌ {error_msg}")
            
    except Exception as e:
        print(f"Error handling client request: {e}")
        import traceback
        traceback.print_exc()
        client_socket.send(f"ERROR: Error handling request\n".encode('utf-8'))
    finally:
        try:
            client_socket.close()
            print(f"🔒 Client {address} connection closed")
        except:
            pass

class ImagePathHandler(BaseHTTPRequestHandler):
    """HTTP request handler for updating image paths"""
    
    def do_GET(self):
        """Handle GET requests to return current image path"""
        if self.path == '/image-path':
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            with image_path_lock:
                response = {
                    'status': 'success',
                    'image_path': current_image_path,
                    'exists': os.path.exists(current_image_path)
                }
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def do_POST(self):
        """Handle POST requests to update image path"""
        if self.path == '/image-path':
            try:
                # Get request body length
                content_length = int(self.headers.get('Content-Length', 0))
                
                # Read request body
                post_data = self.rfile.read(content_length).decode('utf-8')
                print(f"📥 Received POST data: {post_data}")
                
                # Try multiple parsing methods
                new_path = None
                
                # Method 1: Parse URL-encoded form data
                try:
                    parsed_data = urllib.parse.parse_qs(post_data)
                    new_path = parsed_data.get('path', [None])[0]
                    if new_path:
                        print(f"📝 Method 1 successfully parsed path: {new_path}")
                except:
                    pass
                
                # Method 2: Directly parse "path=VALUE" format
                if not new_path and post_data.startswith('path='):
                    new_path = post_data[5:]  # Remove "path=" prefix
                    print(f"📝 Method 2 successfully parsed path: {new_path}")
                
                # Method 3: Handle JSON format
                if not new_path:
                    try:
                        json_data = json.loads(post_data)
                        new_path = json_data.get('path')
                        if new_path:
                            print(f"📝 Method 3 successfully parsed path: {new_path}")
                    except:
                        pass
                
                if new_path:
                    # Update global image path
                    global current_image_path
                    with image_path_lock:
                        current_image_path = new_path
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json; charset=utf-8')
                    self.end_headers()
                    response = {
                        'status': 'success',
                        'message': f'Image path updated to: {new_path}',
                        'path': new_path
                    }
                    print(f"📝 Image path updated to: {new_path}")
                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json; charset=utf-8')
                    self.end_headers()
                    response = {
                        'status': 'error',
                        'message': 'Missing path parameter'
                    }
                
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                response = {
                    'status': 'error',
                    'message': f'Error processing request: {str(e)}'
                }
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        """Override log method to avoid console output"""
        pass

def start_http_server(host='localhost', port=8080, stop_event=None):
    """Start HTTP server"""
    try:
        http_server = HTTPServer((host, port), ImagePathHandler)
        print(f"🌐 HTTP server started at {host}:{port}")
        print(f"   Query current path: curl http://{host}:{port}/image-path")
        print(f"   Update image path:")
        print(f"     Direct format: curl -d \"path=C:\\path\\to\\image.jpg\" http://{host}:{port}/image-path")
        print(f"     JSON format: curl -d '{{\"path\": \"C:\\path\\to\\image.jpg\"}}' http://{host}:{port}/image-path")
        
        # Set timeout to periodically check stop_event
        http_server.timeout = 0.5
        while not (stop_event and stop_event.is_set()):
            http_server.handle_request()
        
        print("🌐 HTTP server stopped")
    except Exception as e:
        print(f"HTTP server failed to start: {e}")

def start_server(host='localhost', port=12345, stop_event=None):
    """
    Start TCP server
    
    Args:
        host (str): Server host address
        port (int): Server port
        stop_event: Stop event
    """
    # Create TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Set socket timeout to periodically check stop_event
    server_socket.settimeout(1.0)
    
    try:
        # Bind address and port
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"🚀 Server started at {host}:{port}")
        print("🔧 Only handles lastimage command")
        print("⏳ Waiting for client connections...")
        print("💡 Please use client to connect and test")
        
        while not (stop_event and stop_event.is_set()):
            try:
                # Accept client connection (with timeout)
                client_socket, address = server_socket.accept()
                
                # Create new thread for each client
                client_thread = threading.Thread(
                    target=handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
                
            except socket.timeout:
                # Timeout exception, continue loop to check stop_event
                continue
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            server_socket.close()
        except:
            pass

if __name__ == "__main__":
    # Default configuration
    TCP_HOST = 'localhost'
    TCP_PORT = 12345
    HTTP_HOST = 'localhost'
    HTTP_PORT = 8080
    
    print("=" * 50)
    print("🔧 Openterface Server Simulator")
    print("=" * 50)
    print("Features:")
    print("  - Simulates TCP server")
    print("  - Handles lastimage command")
    print("  - Supports dynamic image path updates")
    print("  - Provides HTTP interface for path management")
    print("  - Supports multiple data formats (direct, URL-encoded, JSON)")
    print("  - UTF-8 encoding support for proper Chinese character display")
    print("-")
    print("TCP Server:")
    print(f"  - Address: {TCP_HOST}:{TCP_PORT}")
    print("  - Command: lastimage")
    print("-")
    print("HTTP Server:")
    print(f"  - Address: http://{HTTP_HOST}:{HTTP_PORT}")
    print("  - Query path: curl http://localhost:8080/image-path")
    print("  - Update path:")
    print("    - Direct format: curl -d \"path=C:\\path\\to\\image.jpg\" http://localhost:8080/image-path")
    print("    - JSON format: curl -d '{\"path\": \"C:\\path\\to\\image.jpg\"}' http://localhost:8080/image-path")
    print("-")
    print("Current image path:")
    print(f"  - {current_image_path}")
    print("-")
    
    try:
        # Create stop event to notify servers to stop
        stop_event = threading.Event()
        
        # Start HTTP server in new thread
        http_thread = threading.Thread(
            target=start_http_server,
            args=(HTTP_HOST, HTTP_PORT, stop_event),
            daemon=True
        )
        http_thread.start()
        
        # Start TCP server in main thread
        start_server(TCP_HOST, TCP_PORT, stop_event)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping all servers...")
        # Set stop event to notify servers to stop
        stop_event.set()
        print("✅ All servers stopped")
    except Exception as e:
        print(f"Server error: {e}")
        import traceback
        traceback.print_exc()
