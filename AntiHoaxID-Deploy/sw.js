const CACHE = 'antihoaxid-v1';
const ASSETS = [
  './',
  './index.html',
  './style.css',
  './app.js',
  './model.json',
  './Logo.png',
  './Logo Ugm.png',
  './Logo DIKE.jfif',
  './Logo DCSE.png',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  // Biarkan request ke API eksternal (Serper, Google) lewat jaringan langsung
  if (!e.request.url.startsWith(self.location.origin)) return;
  e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request))
  );
});
