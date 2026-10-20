import { sendPush } from './webpush.js';

/**
 * Flag Predictor alerts: stores Web Push subscriptions for the site, and every minute (cron)
 * checks the official OURCs flags and alerts everyone signed up for a stretch the moment its
 * flag changes.
 *
 *   GET  /vapid        -> { publicKey }      (the VAPID public key, base64url)
 *   POST /subscribe    { subscription, rivers: ["isis", "godstow"] }
 *   POST /unsubscribe  { endpoint }
 *   GET  /flags        -> { isis: {flag, label, set_at}, godstow: {...}, fetched_at }
 *                         (the official flags now, for the page's flag tile; at most ~10 s old)
 *
 * Subscriptions live in the SUBS KV namespace as "sub:<sha-256 of the endpoint>". The forecast
 * alerts ("likely to change") are sent by the hourly forecast job (push/send_push.py); the
 * "flag has changed" alerts by this Worker's scheduled handler, so they arrive within a minute.
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

// ---- official flag watch (cron, every minute) ----
const NAMES = { isis: 'Isis', godstow: 'Godstow' };
const LETTERS = { G: 'green', LB: 'light_blue', DB: 'dark_blue', B: 'blue', A: 'amber', R: 'red', K: 'black',
  BL: 'black', X: 'grey', GR: 'grey', GY: 'grey' };
const SITE_URL = 'https://flagpredictor.com/';
const EMOJI = { green: '🟢', light_blue: '🔷', blue: '🔵', dark_blue: '🔵', amber: '🟠', red: '🔴', black: '⚫', grey: '⚪' };
const SEVERITY = ['green', 'light_blue', 'blue', 'dark_blue', 'amber', 'red', 'black'];   // grey: not monitored, no order
/** "⬇️ 🟠 " for a change down to Amber: an arrow (if both flags have an order) and the new flag's colour. */
export function changeMark(from, to) {
  const a = SEVERITY.indexOf(from), b = SEVERITY.indexOf(to);
  const arrow = a >= 0 && b >= 0 && a !== b ? (b > a ? '⬆️ ' : '⬇️ ') : '';
  return arrow + (EMOJI[to] ? EMOJI[to] + ' ' : '');
}

export function flagFromStatus(j) {
  const text = String((j && j.status_text) || '').trim();
  if (text) return { key: text.toLowerCase().replace(/\s+/g, '_').replace('gray', 'grey'), label: text };
  const key = LETTERS[String((j && j.status) || '').toUpperCase()];
  return key ? { key, label: key.replace('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase()) } : null;
}

async function subscriptions(env) {
  const out = [];
  let cursor;
  do {
    const page = await env.SUBS.list({ prefix: 'sub:', cursor });
    for (const k of page.keys) {
      const raw = await env.SUBS.get(k.name);
      if (raw) { try { out.push({ key: k.name, ...JSON.parse(raw) }); } catch (e) { /* skip */ } }
    }
    cursor = page.list_complete ? undefined : page.cursor;
  } while (cursor);
  return out;
}

export async function checkFlags(env, fetchStatus = (r) => fetch(`https://ourcs.co.uk/api/flags/status/${r}/`, {
  headers: { 'User-Agent': 'FlagPredictor alerts (flagpredictor.com)' }, signal: AbortSignal.timeout(10000),
})) {
  const changes = [];
  for (const river of RIVERS) {
    let flag;
    try {
      const res = await fetchStatus(river);
      if (!res.ok) continue;
      const j = await res.json();
      flag = flagFromStatus(j);
      if (flag) flag.setAt = j.set_date || null;
    } catch (e) { continue; }                 // OURCs unreachable: try again next minute
    if (!flag) continue;
    const stateKey = `flag:${river}`;
    const prev = JSON.parse((await env.SUBS.get(stateKey)) || 'null');
    if (prev && prev.key === flag.key) continue;
    await env.SUBS.put(stateKey, JSON.stringify(flag));    // first, so the next minute does not send it again
    if (prev) changes.push({ river, flag, prev });
  }
  if (!changes.length) return [];
  const [privatePem, publicKey] = await Promise.all([env.SUBS.get('config:vapid_private'), env.SUBS.get('config:vapid_public')]);
  if (!privatePem || !publicKey) return changes;
  const vapid = { privatePem, publicKey, subject: 'https://flagpredictor.com' };
  const subs = await subscriptions(env);
  for (const { river, flag, prev } of changes) {
    const payload = {
      title: 'Flag Predictor',
      body: `${changeMark(prev.key, flag.key)}${NAMES[river]} Flag now ${flag.label}.`,
      url: SITE_URL,
      // one tag per change, shown without a second buzz if it ever arrives twice
      tag: `flag-change-${river}-${flag.setAt || flag.key}`,
      renotify: false,
    };
    for (const s of subs) {
      if (!(s.rivers || []).includes(river)) continue;
      try {
        const status = await sendPush(s.subscription, payload, vapid);
        if (status === 404 || status === 410) await env.SUBS.delete(s.key);
      } catch (e) { /* one bad subscription must not stop the rest */ }
    }
  }
  return changes;
}

// ---- the official flags now, for the page (GET /flags) ----
// OURCs does not answer a browser on another site, so the page asks here. Each Worker
// instance keeps the answer for a few seconds, so many viewers do not each hit OURCs.
const FLAGS_FRESH_MS = 8000;
let flagsCache = null;

export async function currentFlags(fetchStatus = (r) => fetch(`https://ourcs.co.uk/api/flags/status/${r}/`, {
  headers: { 'User-Agent': 'FlagPredictor (flagpredictor.com)' }, signal: AbortSignal.timeout(5000),
}), now = Date.now()) {
  if (flagsCache && now - flagsCache.at < FLAGS_FRESH_MS) return flagsCache.body;
  const body = { fetched_at: new Date(now).toISOString() };
  await Promise.all(RIVERS.map(async (river) => {
    try {
      const res = await fetchStatus(river);
      if (!res.ok) return;
      const j = await res.json();
      const f = flagFromStatus(j);
      if (f) body[river] = { flag: f.key, label: f.label, set_at: j.set_date || null };
    } catch (e) { /* OURCs unreachable: leave this stretch out */ }
  }));
  if (RIVERS.some((r) => body[r])) flagsCache = { at: now, body };
  return body;
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(checkFlags(env));
  },

  async fetch(request, env) {
    const origin = request.headers.get('Origin');
    const { pathname } = new URL(request.url);
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors(origin) });

    if (request.method === 'GET' && pathname === '/flags') {
      const r = json(await currentFlags(), 200, origin);
      r.headers.set('Cache-Control', 'no-store');
      return r;
    }

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
