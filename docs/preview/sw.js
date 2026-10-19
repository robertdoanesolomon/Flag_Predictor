/* Flag Predictor service worker: shows flag alerts (push/send_push.py) and opens the page on tap.
   No offline caching: the forecast changes every hour. */

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('push', (event) => {
  let msg = {};
  try { msg = event.data ? event.data.json() : {}; } catch (e) { msg = { body: event.data ? event.data.text() : '' }; }
  const url = msg.url || self.registration.scope;
  event.waitUntil(self.registration.showNotification(msg.title || 'Flag Predictor', {
    body: msg.body || '',
    tag: msg.tag || 'flag',
    renotify: true,
    icon: 'icons/icon-192.png',
    badge: 'icons/badge-72.png',
    data: { url },
  }));
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const url = (event.notification.data && event.notification.data.url) || self.registration.scope;
  event.waitUntil((async () => {
    const wins = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    for (const w of wins) {
      if (w.url.startsWith(self.registration.scope) && 'focus' in w) {
        try { await w.navigate(url); } catch (e) { /* same page */ }
        return w.focus();
      }
    }
    return self.clients.openWindow(url);
  })());
});
