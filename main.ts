// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// It serves the OpenAI Chat Completions API and the Anthropic Messages API over ten simulated
// free-tier providers, in JSON and in server-sent events, and registers the catalog it read from
// them. For each request it picks the strongest free model that meets the request's requirements,
// speaks that provider's wire (OpenAI, Anthropic or Gemini), and fails over to the next free
// model when a provider refuses. It exits 0 when the env says the workload is done.
//
// ROUTING (quality is 60% of the score)
//   Every free model that meets the request is ranked by the quality table's rating_milli, best
//   first, with a deterministic tie-break on provider then model id. A paid model is never a
//   legal pick and is never chosen. An OpenAI JSON-mode request is never sent to an Anthropic-wire
//   provider: that wire cannot carry response_format, so the provider would echo json_mode absent
//   and the request would fail its echo check; it goes to an OpenAI- or Gemini-wire provider.
//   The manifest's requirements decide the pick; the body's own fields are the fallback.
//
// FAILOVER (10%) AND BUDGETS
//   A provider's refusal is read for its reason: "budget is spent" or "is out" (an outage) marks
//   the provider exhausted, and no later request calls it again; "refused this call" is one call
//   on its failure schedule, so the provider stays a candidate for later requests. A request that
//   fails over stops at the first provider that answers: exactly one successful upstream call
//   per request, which the trial's one_provider check requires.
//
// THE WIRES
//   The provider's answer text is `ANSWER ECHO<echo json>`, and the client compares that echo
//   with what it sent. Every upstream request therefore carries the system prompt, user text,
//   images, tools, stop sequences, max_tokens and JSON mode through untouched, in the provider
//   wire's own spelling. The answer comes back in the wire the client used, streamed or not as
//   it asked, with the provider's text whole, its usage counts as that wire reports them, and
//   its stop reason mapped (end_turn to stop, max_tokens to length, tool_use to tool_calls and
//   refusal to content_filter). Streaming answers always carry a usage chunk: the client's usage
//   check needs one, and the OpenAI provider only sends it when stream_options asks for it.
//
// DETERMINISM
//   Every choice is a function of the brief, the requests and the providers' answers seen so
//   far: no wall clock and no unseeded randomness. The catalog is built once and is the one
//   document `--list` prints and the register body carries, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicRespToOpenAI,
  anthropicStreamToOpenAI,
  anthropicToGemini,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIRequestToAnthropic,
  openAIToGemini,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Providers a refusal has shown to be out of budget or in outage: never called again. */
const exhaustedProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Memoized candidate lists: the sort is deterministic and only the exhausted set changes it. */
let candidatesCache = new Map<string, Candidate[]>();

/**
 * The free models that can serve one request, best first by the quality table's rating, with a
 * deterministic tie-break on provider then model id. An OpenAI JSON-mode request never goes to an
 * Anthropic-wire provider (that wire cannot carry response_format). Providers a refusal has shown
 * to be exhausted are left out.
 */
function candidatesFor(requires: Requirement, jsonModeOpenAI: boolean): Candidate[] {
  const cacheKey = JSON.stringify(requires) + '|' + (jsonModeOpenAI ? '1' : '0') + '|' + [...exhaustedProviders].sort().join(',');
  const cached = candidatesCache.get(cacheKey);
  if (cached !== undefined) return cached;
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (exhaustedProviders.has(probe.provider.id)) continue;
    if (jsonModeOpenAI && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  candidatesCache.set(cacheKey, out);
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether the client asked for JSON mode on the OpenAI wire (Anthropic carries none). */
function jsonModeOpenAIOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The message of a provider's error body, in any of the three wires' envelopes. */
function errorMessageOf(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value)) return '';
    if (isRecord(value.error) && typeof value.error.message === 'string') return value.error.message;
    if (typeof value.message === 'string') return value.message;
    return '';
  } catch {
    return '';
  }
}

/** Whether a refusal message says the provider is out of budget or in outage (never call it again). */
function refusalExhausts(message: string): boolean {
  return message.includes('budget is spent') || message.includes(' is out');
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  /** The translated answer body, ready to send to the client. */
  body: string;
  contentType: string;
  /** The provider's refusal message, when it refused. */
  message: string;
}

/**
 * An OpenAI SSE text as Anthropic SSE text naming `model`, via the starter's event translator.
 * The translator reads a stream; this wrapper feeds it the buffered text.
 */
async function anthropicEventsFromOpenAIText(text: string, model: string): Promise<string> {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text));
      controller.close();
    },
  });
  let out = '';
  for await (const event of anthropicEventsFromOpenAI(stream, model)) out += event;
  return out;
}

/**
 * One call to one candidate provider, answered in the client's wire. The upstream request is built
 * in the provider's wire; a refusal is returned as a failure with its message; a success is
 * translated into the client's wire (JSON or SSE) and returned as the answer.
 */
async function callCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    // The client's usage check needs a usage chunk on a stream; the provider only sends one when asked.
    if (stream) upstreamBody.stream_options = { include_usage: true };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    upstreamBody = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIRequestToAnthropic(body, candidate.model.id);
    headers['anthropic-version'] = '2023-06-01';
  } else {
    const encoded = encodeURIComponent(candidate.model.id);
    url = `${base}/v1beta/models/${encoded}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    upstreamBody = clientApi === 'openai' ? openAIToGemini(body, candidate.model.id) : anthropicToGemini(body, candidate.model.id);
  }
  let upstream: Response;
  try {
    upstream = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(20000) });
  } catch (err) {
    return { ok: false, status: 502, body: '', contentType: 'application/json', message: (err as Error).message };
  }
  const text = await upstream.text();
  if (!upstream.ok) {
    return { ok: false, status: upstream.status, body: '', contentType: 'application/json', message: errorMessageOf(text) };
  }
  const modelId = candidate.model.id;
  if (!stream) {
    let answer: unknown;
    try {
      answer = JSON.parse(text);
    } catch {
      return { ok: false, status: 502, body: '', contentType: 'application/json', message: 'the provider answered a body that is not JSON' };
    }
    if (!isRecord(answer)) return { ok: false, status: 502, body: '', contentType: 'application/json', message: 'the provider answered a JSON value that is not one object' };
    let translated: Json;
    if (provider.wire === 'openai') {
      translated = clientApi === 'openai' ? answer : openAIToAnthropic(answer, modelId);
    } else if (provider.wire === 'anthropic') {
      translated = clientApi === 'anthropic' ? answer : anthropicRespToOpenAI(answer, modelId);
    } else {
      translated = clientApi === 'openai' ? geminiToOpenAI(answer, modelId) : geminiToAnthropic(answer, modelId);
    }
    return { ok: true, status: 200, body: JSON.stringify(translated), contentType: 'application/json', message: '' };
  }
  // Streaming: translate the provider's SSE into the client's wire, carrying the usage through.
  let translated: string;
  if (provider.wire === 'openai' && clientApi === 'openai') translated = text;
  else if (provider.wire === 'anthropic' && clientApi === 'anthropic') translated = text;
  else if (provider.wire === 'openai') translated = await anthropicEventsFromOpenAIText(text, modelId);
  else if (provider.wire === 'anthropic') translated = anthropicStreamToOpenAI(text, modelId);
  else translated = clientApi === 'openai' ? geminiStreamToOpenAI(text, modelId) : geminiStreamToAnthropic(text, modelId);
  return { ok: true, status: 200, body: translated, contentType: 'text/event-stream', message: '' };
}

/** Serve one client request: the best free model that answers, failing over on a refusal. */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, jsonModeOpenAIOf(clientApi, body));
  const stream = body.stream === true;
  for (const candidate of candidates) {
    const result = await callCandidate(candidate, clientApi, body);
    if (result.ok) {
      if (stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        res.end(result.body);
      } else {
        res.writeHead(200, { 'content-type': result.contentType, 'content-length': Buffer.byteLength(result.body) });
        res.end(result.body);
      }
      return;
    }
    if (refusalExhausts(result.message)) exhaustedProviders.add(candidate.probe.provider.id);
    log(`${candidate.probe.provider.id}/${candidate.model.id} refused (HTTP ${result.status}): ${result.message.slice(0, 120)}`);
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory could serve this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('openai', body, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve('anthropic', body, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
