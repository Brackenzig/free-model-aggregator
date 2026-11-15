// The two upstream wires the starter does not speak: Gemini and Anthropic as a provider.
//
// The entry's client is OpenAI (Chat Completions) or Anthropic (Messages). Its providers speak
// OpenAI, Anthropic or Gemini. This file builds the upstream request for a Gemini or an Anthropic
// provider from either client body, and translates a Gemini or Anthropic provider answer back into
// either client wire, in JSON and in server-sent events.
//
// The provider's answer text is always `ANSWER ECHO<echo json>`: the echo is what the client
// compares against what it sent, so every request builder below carries the client's system prompt,
// user text, images, tools, stop sequences, max_tokens and JSON mode through untouched, in the
// provider wire's own spelling. A field rewritten or dropped loses the request.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same
// body and the same provider answer give the same bytes on a replay.

import { isRecord, type Json } from './providers.ts';
import { sseEvent, stopReasonOf } from './wire.ts';

/** A fixed epoch second for every translated OpenAI reply's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of an Anthropic system value (a string or an array of text blocks). */
function systemTextOf(system: unknown): string {
  return textOf(system);
}

/** One OpenAI message's content as Gemini parts: text and image_url blocks. */
function openAIParts(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const url = block.image_url.url;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

/** One Anthropic content block as a Gemini part: text or an inline image; null for the rest. */
function anthropicPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** An OpenAI tool definition as a Gemini function declaration (name first: the echo reads names). */
function openAIToolDecl(raw: unknown): Json | null {
  if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return null;
  const fn = raw.function;
  return {
    name: fn.name,
    ...(typeof fn.description === 'string' ? { description: fn.description } : {}),
    ...(isRecord(fn.parameters) ? { parameters: fn.parameters } : {}),
  };
}

/** An Anthropic tool definition as a Gemini function declaration. */
function anthropicToolDecl(raw: unknown): Json | null {
  if (!isRecord(raw) || typeof raw.name !== 'string') return null;
  return {
    name: raw.name,
    ...(typeof raw.description === 'string' ? { description: raw.description } : {}),
    ...(isRecord(raw.input_schema) ? { parameters: raw.input_schema } : {}),
  };
}

/** The generation config of a Gemini request: the output cap, stop sequences, JSON mode and sampling. */
function generationConfig(maxTokens: unknown, stops: unknown, jsonMode: boolean, temperature: unknown, topP: unknown): Json {
  const config: Json = {};
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (Array.isArray(stops) && stops.length > 0) config.stopSequences = stops.filter((s) => typeof s === 'string');
  else if (typeof stops === 'string' && stops !== '') config.stopSequences = [stops];
  if (jsonMode) config.responseMimeType = 'application/json';
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request naming `model` (the model
 * travels in the URL, not the body): the system prompt becomes systemInstruction, user and assistant
 * turns become contents, images become inlineData or fileData, tool definitions become
 * functionDeclarations, and max_tokens, stop sequences, JSON mode and sampling pass through.
 */
export function openAIToGemini(body: Json, model: string): Json {
  const out: Json = {};
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const contents: Json[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
        let args: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        } else if (isRecord(call.function.arguments)) args = call.function.arguments;
        parts.push({ functionCall: { name: call.function.name, args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      const part: Json = { functionResponse: { name: typeof raw.name === 'string' ? raw.name : '', response: text !== '' ? { content: text } : {} } };
      if (typeof raw.tool_call_id === 'string' && raw.tool_call_id !== '') part.functionResponse = { name: raw.tool_call_id, response: text !== '' ? { content: text } : {} };
      contents.push({ role: 'user', parts: [part] });
      continue;
    }
    const parts = openAIParts(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const systemText = systemParts.join('\n');
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((t) => {
      const d = openAIToolDecl(t);
      return d === null ? [] : [d];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const jsonMode = format === 'json_object' || format === 'json_schema';
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  out.generationConfig = generationConfig(maxTokens, body.stop, jsonMode, body.temperature, body.top_p);
  void model;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request: the system value becomes
 * systemInstruction, user and assistant turns become contents, image blocks become inlineData or
 * fileData, tool_use blocks become functionCall parts, tool_result blocks become functionResponse
 * parts, and max_tokens, stop sequences and sampling pass through (Anthropic carries no JSON mode).
 */
export function anthropicToGemini(body: Json, model: string): Json {
  const out: Json = {};
  const systemText = systemTextOf(body.system);
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (assistant && block.type === 'tool_use') {
        parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        continue;
      }
      if (!assistant && block.type === 'tool_result') {
        const text = typeof block.content === 'string' ? block.content : textOf(block.content);
        parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: text !== '' ? { content: text } : {} } });
        continue;
      }
      const part = anthropicPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((t) => {
      const d = anthropicToolDecl(t);
      return d === null ? [] : [d];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences : undefined;
  out.generationConfig = generationConfig(body.max_tokens, stops, false, body.temperature, body.top_p);
  void model;
  return out;
}

/** The text of a Gemini response: every candidate part's text, concatenated. */
export function geminiTextOf(resp: Json): string {
  const candidates = Array.isArray(resp.candidates) ? resp.candidates : [];
  const parts: string[] = [];
  for (const candidate of candidates) {
    if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) continue;
    for (const part of candidate.content.parts) {
      if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
    }
  }
  return parts.join('');
}

/** The usage of a Gemini response: { input, output } token counts, 0 when absent. */
export function geminiUsageOf(resp: Json): { input: number; output: number } {
  const meta = isRecord(resp.usageMetadata) ? resp.usageMetadata : {};
  const num = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
  return { input: num(meta.promptTokenCount), output: num(meta.candidatesTokenCount) };
}

/** A Gemini finish reason as an OpenAI finish_reason. */
function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finish reason as an Anthropic stop_reason. */
function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The first candidate's finish reason of a Gemini response, or null. */
function geminiFinishOf(resp: Json): unknown {
  const candidates = Array.isArray(resp.candidates) ? resp.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  return first === null ? null : first.finishReason ?? null;
}

/** A Gemini JSON answer as an OpenAI Chat Completions object naming `model`. */
export function geminiToOpenAI(resp: Json, model: string): Json {
  const usage = geminiUsageOf(resp);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiTextOf(resp) }, finish_reason: geminiStopToOpenAI(geminiFinishOf(resp)), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini JSON answer as an Anthropic Messages object naming `model`. */
export function geminiToAnthropic(resp: Json, model: string): Json {
  const usage = geminiUsageOf(resp);
  const text = geminiTextOf(resp);
  return {
    id: `msg_${model}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiStopToAnthropic(geminiFinishOf(resp)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** The text of an Anthropic JSON answer: its text blocks, concatenated. */
export function anthropicRespTextOf(resp: Json): string {
  if (!Array.isArray(resp.content)) return '';
  return resp.content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** The usage of an Anthropic JSON answer: { input, output } token counts, 0 when absent. */
export function anthropicRespUsageOf(resp: Json): { input: number; output: number } {
  const usage = isRecord(resp.usage) ? resp.usage : {};
  const num = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
  return { input: num(usage.input_tokens), output: num(usage.output_tokens) };
}

/** An Anthropic JSON answer as an OpenAI Chat Completions object naming `model`. */
export function anthropicRespToOpenAI(resp: Json, model: string): Json {
  const usage = anthropicRespUsageOf(resp);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: anthropicRespTextOf(resp) }, finish_reason: stopReasonOf(resp.stop_reason) === 'end_turn' ? 'stop' : stopReasonOf(resp.stop_reason) === 'max_tokens' ? 'length' : stopReasonOf(resp.stop_reason) === 'tool_use' ? 'tool_calls' : stopReasonOf(resp.stop_reason) === 'refusal' ? 'content_filter' : 'stop', logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request: the system messages become
 * the system value, user parts become text and image blocks, assistant tool calls become tool_use
 * blocks, tool messages become tool_result blocks, and max_tokens, stop sequences, tools, tool
 * choice and sampling pass through. The provider's echo reads the system text, user text, image,
 * tool names, max_tokens and stop sequences off this shape, so each is carried through unchanged.
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        } else if (isRecord(call.function.arguments)) input = call.function.arguments;
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: call.function.name, input });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: text }] });
      continue;
    }
    const blocks: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  const systemText = systemParts.join('\n');
  if (systemText !== '') out.system = systemText;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : [];
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** One server-sent event frame: its event name (null when unnamed) and its data payload. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** The frames of an SSE text: split on blank lines, `event:` and `data:` lines kept. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** A JSON value of one SSE frame's data, or null when it is not JSON. */
function frameJson(frame: SseFrame): Json | null {
  try {
    const value: unknown = JSON.parse(frame.data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One OpenAI chat chunk naming `model`: a delta, a finish_reason, or an empty usage chunk. */
function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] };
}

/** One OpenAI chat usage chunk naming `model`. */
function openAIUsageChunk(model: string, input: number, output: number): Json {
  return { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output } };
}

/**
 * A Gemini SSE answer as OpenAI SSE text naming `model`: the role chunk, one content chunk per text
 * piece, the finish chunk, the usage chunk the client checks for, then [DONE]. The Gemini frames
 * carry text pieces and, on the last frame, the finish reason and usageMetadata.
 */
export function geminiStreamToOpenAI(text: string, model: string): string {
  const pieces: string[] = [];
  let usage = { input: 0, output: 0 };
  let finish: unknown = null;
  for (const frame of parseSseFrames(text)) {
    const value = frameJson(frame);
    if (value === null) continue;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    for (const candidate of candidates) {
      if (!isRecord(candidate)) continue;
      if (isRecord(candidate.content) && Array.isArray(candidate.content.parts)) {
        for (const part of candidate.content.parts) {
          if (isRecord(part) && typeof part.text === 'string' && part.text !== '') pieces.push(part.text);
        }
      }
      if (candidate.finishReason !== undefined) finish = candidate.finishReason;
    }
    if (isRecord(value.usageMetadata)) usage = geminiUsageOf(value);
  }
  const events: string[] = [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null))];
  for (const piece of pieces) events.push(sseEvent(null, openAIChunk(model, { content: piece }, null)));
  events.push(sseEvent(null, openAIChunk(model, {}, geminiStopToOpenAI(finish))));
  events.push(sseEvent(null, openAIUsageChunk(model, usage.input, usage.output)));
  events.push('data: [DONE]\n\n');
  return events.join('');
}

/**
 * A Gemini SSE answer as Anthropic SSE text naming `model`: message_start with the input tokens,
 * one text content block with a delta per piece, content_block_stop, message_delta with the stop
 * reason and output tokens, then message_stop.
 */
export function geminiStreamToAnthropic(text: string, model: string): string {
  const pieces: string[] = [];
  let usage = { input: 0, output: 0 };
  let finish: unknown = null;
  for (const frame of parseSseFrames(text)) {
    const value = frameJson(frame);
    if (value === null) continue;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    for (const candidate of candidates) {
      if (!isRecord(candidate)) continue;
      if (isRecord(candidate.content) && Array.isArray(candidate.content.parts)) {
        for (const part of candidate.content.parts) {
          if (isRecord(part) && typeof part.text === 'string' && part.text !== '') pieces.push(part.text);
        }
      }
      if (candidate.finishReason !== undefined) finish = candidate.finishReason;
    }
    if (isRecord(value.usageMetadata)) usage = geminiUsageOf(value);
  }
  const events: string[] = [
    sseEvent('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } } }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of pieces) events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  events.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: geminiStopToAnthropic(finish), stop_sequence: null }, usage: { output_tokens: usage.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return events.join('');
}

/**
 * An Anthropic SSE answer as OpenAI SSE text naming `model`: the role chunk, one content chunk per
 * text delta, the finish chunk, the usage chunk the client checks for, then [DONE]. The Anthropic
 * frames carry the input tokens in message_start and the output tokens in message_delta.
 */
export function anthropicStreamToOpenAI(text: string, model: string): string {
  const pieces: string[] = [];
  let input = 0;
  let output = 0;
  let stop: string | null = null;
  for (const frame of parseSseFrames(text)) {
    if (frame.event === null) continue;
    const value = frameJson(frame);
    if (value === null) continue;
    if (frame.event === 'message_start' && isRecord(value.message) && isRecord(value.message.usage)) {
      const usage = value.message.usage;
      if (typeof usage.input_tokens === 'number') input = usage.input_tokens;
    } else if (frame.event === 'content_block_delta' && isRecord(value.delta) && value.delta.type === 'text_delta' && typeof value.delta.text === 'string' && value.delta.text !== '') {
      pieces.push(value.delta.text);
    } else if (frame.event === 'message_delta') {
      if (isRecord(value.delta) && typeof value.delta.stop_reason === 'string') stop = value.delta.stop_reason;
      if (isRecord(value.usage) && typeof value.usage.output_tokens === 'number') output = value.usage.output_tokens;
    }
  }
  const finish = stop === null ? 'stop' : stopReasonOf(stop) === 'end_turn' ? 'stop' : stopReasonOf(stop) === 'max_tokens' ? 'length' : stopReasonOf(stop) === 'tool_use' ? 'tool_calls' : stopReasonOf(stop) === 'refusal' ? 'content_filter' : 'stop';
  const events: string[] = [sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null))];
  for (const piece of pieces) events.push(sseEvent(null, openAIChunk(model, { content: piece }, null)));
  events.push(sseEvent(null, openAIChunk(model, {}, finish)));
  events.push(sseEvent(null, openAIUsageChunk(model, input, output)));
  events.push('data: [DONE]\n\n');
  return events.join('');
}
