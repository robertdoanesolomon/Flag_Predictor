/**
 * Flag Predictor alerts: stores Web Push subscriptions for the site.
 *
 *   GET  /vapid        -> { publicKey }      (the VAPID public key, base64url)
 *   POST /subscribe    { subscription, rivers: ["isis", "godstow"] }
 *   POST /unsubscribe  { endpoint }
 *
 * Subscriptions live in the SUBS KV namespace as "sub:<sha-256 of the endpoint>".
 * The hourly forecast job (send_push.py) reads them, and the VAPID private key, through the
 * Cloudflare API and sends the notifications; this Worker never sends anything.
 */

const RIVERS = ['isis', 'godstow'];
const ALLOWED_ORIGINS = [
  'https://flagpredictor.com',
  'https://www.flagpredictor.com',
  'https://robertdoanesolomon.github.io',
];
// Push services a browser subscription can point at (Chrome/Android, Firefox, Safari/iOS, Edge).
const PUSH_HOSTS = [/\.googleapis\.com$/, /\.mozilla\.com$/, /\.push\.apple\.com$/, /\.notify\.windows\.com$/, /\.mozaws\.net$/];

function cors(origin) {
  const ok = ALLOWED_ORIGINS.includes(origin) || /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin || '');
  return {
    'Access-Control-Allow-Origin': ok ? origin : ALLOWED_ORIGINS[0],
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    Vary: 'Origin',
  };
}

function json(body, status, origin) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...cors(origin) },
  });
}

async function keyFor(endpoint) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(endpoint));
  return 'sub:' + [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function validSubscription(s) {
  if (!s || typeof s.endpoint !== 'string' || s.endpoint.length > 1000) return false;
  let url;
  try { url = new URL(s.endpoint); } catch { return false; }
  if (url.protocol !== 'https:' || !PUSH_HOSTS.some((re) => re.test(url.hostname))) return false;
  const k = s.keys || {};
  return typeof k.p256dh === 'string' && typeof k.auth === 'string' && k.p256dh.length < 200 && k.auth.length < 100;
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin');
    const { pathname } = new URL(request.url);
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors(origin) });

    if (request.method === 'GET' && pathname === '/vapid') {
      const publicKey = await env.SUBS.get('config:vapid_public');
      return publicKey ? json({ publicKey }, 200, origin) : json({ error: 'not set up yet' }, 503, origin);
    }

    if (request.method === 'POST' && (pathname === '/subscribe' || pathname === '/unsubscribe')) {
      let body;
      try { body = await request.json(); } catch { return json({ error: 'bad json' }, 400, origin); }
      if (pathname === '/unsubscribe') {
        if (!body || typeof body.endpoint !== 'string') return json({ error: 'no endpoint' }, 400, origin);
        await env.SUBS.delete(await keyFor(body.endpoint));
        return json({ ok: true }, 200, origin);
      }
      const sub = body && body.subscription;
      if (!validSubscription(sub)) return json({ error: 'bad subscription' }, 400, origin);
      const rivers = Array.isArray(body.rivers) ? RIVERS.filter((r) => body.rivers.includes(r)) : [];
      if (!rivers.length) return json({ error: 'choose a stretch' }, 400, origin);
      const record = {
        subscription: { endpoint: sub.endpoint, keys: { p256dh: sub.keys.p256dh, auth: sub.keys.auth } },
        rivers,
        created: new Date().toISOString(),
      };
      await env.SUBS.put(await keyFor(sub.endpoint), JSON.stringify(record));
      return json({ ok: true, rivers }, 200, origin);
    }

    return json({ error: 'not found' }, 404, origin);
  },
};
