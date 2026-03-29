"""Start dashboard server with localhost.run tunnel and QR code."""
import subprocess
import socket
import time
import sys
import os
import logging

logger = logging.getLogger(__name__)

WS = '/home/boxbunny/Desktop/doomsday_integration/boxing_robot_ws'
os.chdir(WS)


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return 'localhost'


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        f'{WS}/src/boxbunny_dashboard:'
        f'{WS}/src/boxbunny_core:'
        f'{WS}/install/boxbunny_msgs/local/lib/python3.10/dist-packages:'
        + env.get('PYTHONPATH', '')
    )
    proc = subprocess.Popen(
        [sys.executable, '-c',
         'import uvicorn; '
         'from boxbunny_dashboard.server import create_app; '
         'uvicorn.run(create_app(), host="0.0.0.0", port=8080, '
         'log_level="warning")'],
        env=env,
    )
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError('Server failed to start. Check dependencies.')
    return proc


def start_tunnel() -> tuple[subprocess.Popen, str | None]:
    """Start localhost.run SSH tunnel with retry. Returns (process, url)."""
    for attempt in range(3):
        if attempt > 0:
            print(f'Tunnel retry {attempt + 1}/3...')
            time.sleep(2)

        proc = subprocess.Popen(
            ['ssh', '-o', 'StrictHostKeyChecking=no',
             '-o', 'ServerAliveInterval=30',
             '-o', 'ConnectTimeout=10',
             '-R', '80:localhost:8080', 'nokey@localhost.run'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        deadline = time.time() + 20
        while time.time() < deadline:
            line = proc.stdout.readline().decode('utf-8', errors='ignore')
            if not line:
                break
            if '.lhr.life' in line:
                for word in line.split():
                    if word.startswith('https://'):
                        return proc, word.strip().rstrip(',')

        # Failed this attempt - kill and retry
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

    return None, None


def show_qr(url: str) -> None:
    try:
        import qrcode
        from IPython.display import display, Image as IPImage
        from io import BytesIO
        qr = qrcode.QRCode(version=1, box_size=10, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color='white', back_color='#0A0A0A')
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        print('Scan with your phone:')
        display(IPImage(data=buf.read(), width=350))
    except Exception:
        print(f'Open: {url}')


def main() -> None:
    os.system('fuser -k 8080/tcp 2>/dev/null')
    os.system('pkill -f localhost.run 2>/dev/null')
    time.sleep(1)

    ip = get_local_ip()
    server = start_server()

    print(f'Server running (PID {server.pid})')
    print(f'Local: http://{ip}:8080')
    print()

    tunnel, tunnel_url = start_tunnel()

    print('=' * 50)
    if tunnel_url:
        print(f'PUBLIC URL: {tunnel_url}')
        print(f'(also local: http://{ip}:8080)')
    else:
        print(f'Tunnel failed - using LOCAL only: http://{ip}:8080')
        print('Phone must be on the same WiFi network.')
    print('=' * 50)
    print('Logins: alex/boxing123 | maria/boxing123 '
          '| jake/boxing123 | sarah/coaching123')
    print()

    show_qr(tunnel_url or f'http://{ip}:8080')

    print()
    print('=' * 50)
    print('RUNNING — Press STOP to shut down')
    print('=' * 50)

    try:
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if tunnel:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=3)
            except Exception:
                tunnel.kill()
        server.terminate()
        try:
            server.wait(timeout=3)
        except Exception:
            server.kill()
        os.system('fuser -k 8080/tcp 2>/dev/null')
        os.system('pkill -f localhost.run 2>/dev/null')
        print('\nServer and tunnel stopped.')


if __name__ == '__main__':
    main()
