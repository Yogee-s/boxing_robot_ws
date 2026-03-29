"""Generate a QR code for the dashboard on local network."""
import socket
import io
import qrcode
from IPython.display import display, Image as IPImage, HTML


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())


ip = get_local_ip()
port = 8080
url = f"http://{ip}:{port}"

qr = qrcode.QRCode(version=1, box_size=10, border=2)
qr.add_data(url)
qr.make(fit=True)
img = qr.make_image(fill_color="white", back_color="#0D0D0D")

buf = io.BytesIO()
img.save(buf, format='PNG')
buf.seek(0)

display(HTML(f"""
<div style="background:#0D0D0D;padding:24px;border-radius:16px;
            text-align:center;max-width:400px;margin:0 auto;
            font-family:sans-serif">
    <h2 style="color:#00E676;margin:0 0 8px 0">BoxBunny Dashboard</h2>
    <p style="color:#E0E0E0;margin:0 0 4px 0">
        Scan with your phone camera to open</p>
    <p style="color:#9E9E9E;margin:0 0 16px 0;font-size:13px">
        Your phone must be on the same Wi-Fi network as this device
    </p>
</div>
"""))
display(IPImage(data=buf.read()))
display(HTML(f"""
<div style="text-align:center;font-family:monospace;margin-top:8px">
    <p style="color:#E0E0E0;font-size:18px"><b>{url}</b></p>
    <p style="color:#9E9E9E;font-size:13px">
        Login: <b>alex</b> / boxing123 |
        <b>maria</b> / boxing123 |
        <b>jake</b> / boxing123
    </p>
</div>
"""))
