const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Find local Wi-Fi / LAN IP
function getLocalIP() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      if (iface.family === 'IPv4' && !iface.internal) {
        if (!iface.address.startsWith('169.254')) {
          return iface.address;
        }
      }
    }
  }
  return 'localhost';
}

const localIP = getLocalIP();
const HTTP_PORT = 8000;
const HTTPS_PORT = 8443;

function handleRequest(req, res) {
  let filePath = path.join(__dirname, req.url === '/' ? 'index.html' : req.url.split('?')[0]);
  if (!fs.existsSync(filePath)) {
    filePath = path.join(__dirname, 'index.html');
  }

  const ext = path.extname(filePath).toLowerCase();
  const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.ico': 'image/x-icon'
  };
  const contentType = mimeTypes[ext] || 'text/plain';

  fs.readFile(filePath, (err, content) => {
    if (err) {
      res.writeHead(500);
      res.end('Server Error: ' + err.code);
    } else {
      res.writeHead(200, {
        'Content-Type': contentType,
        'Cache-Control': 'no-cache',
        'Access-Control-Allow-Origin': '*'
      });
      res.end(content);
    }
  });
}

// HTTP Server
const httpServer = http.createServer(handleRequest);
httpServer.listen(HTTP_PORT, '0.0.0.0', () => {
  console.log(`[PITCHBLACK] Plain HTTP server running at:`);
  console.log(`  http://${localIP}:${HTTP_PORT}/`);
  console.log(`  http://localhost:${HTTP_PORT}/`);
});

// HTTPS Server (Needed by mobile Chrome for getUserMedia over Wi-Fi)
const keyPath = path.join(__dirname, 'key.pem');
const certPath = path.join(__dirname, 'cert.pem');

if (fs.existsSync(keyPath) && fs.existsSync(certPath)) {
  const options = {
    key: fs.readFileSync(keyPath),
    cert: fs.readFileSync(certPath)
  };

  const httpsServer = https.createServer(options, handleRequest);
  httpsServer.listen(HTTPS_PORT, '0.0.0.0', () => {
    console.log(`\n[PITCHBLACK] HTTPS server (Secure Context for Android Chrome):`);
    console.log(`  >>> https://${localIP}:${HTTPS_PORT}/ <<<`);
    console.log(`\n(On OnePlus 12R Chrome, tap "Advanced" -> "Proceed" to bypass self-signed warning)\n`);
  });
} else {
  console.log('[Notice] cert.pem or key.pem not found. HTTPS server not started.');
}
