# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Brackenzig/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Brackenzig/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A quality-ranked free-model aggregator: one zero-config server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten free-tier providers, in JSON and in server-sent events. For each request it picks the strongest free model that meets the request's requirements (tools, vision, JSON mode, context length), speaks that provider's wire (OpenAI, Anthropic or Gemini), and fails over to the next free model when a provider refuses. `node main.ts --list` prints the catalog it found: every provider, its models, and which are free.

## Run it

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:8080 MAKEFUN_PROMPT_FILE=prompt.txt node main.ts
```

```sh
node main.ts --list
```

The server listens on a port it picks, registers it with the trial env, serves the workload's requests, and exits 0 when the env says the workload is done. It needs Node 22 and nothing else: the standard library alone, TypeScript run directly.

## How it routes

- **Quality first.** Every free model that meets a request is ranked by the brief's quality table (`rating_milli`), best first, with a deterministic tie-break on provider then model id. A paid model is never a legal pick. An OpenAI JSON-mode request never goes to an Anthropic-wire provider, because that wire cannot carry `response_format`.
- **Failover that learns.** A refusal is read for its reason: "budget is spent" or an outage marks the provider exhausted and it is never called again; a single refused call on a provider's failure schedule leaves the provider a candidate for later requests. A request stops at the first provider that answers, so exactly one upstream call succeeds per request.
- **The wires, faithfully.** Every upstream request carries the system prompt, user text, images, tools, stop sequences, `max_tokens` and JSON mode through untouched, in the provider wire's own spelling, because the provider echoes them back and the client compares that echo with what it sent. Answers come back in the wire the client used, with the provider's text whole, its usage counts as that wire reports them, and its stop reason mapped. Streaming answers always carry a usage chunk.
- **Deterministic.** Every choice is a function of the brief, the requests and the providers' answers seen so far: no wall clock, no unseeded randomness. The same trial run twice leaves the same state export, byte for byte.

## Practice results

Scored locally against the kit's sealed env and scorer on all three practice instances:

| trial | score | quality | served | wire | failover | catalog |
|-------|-------|---------|--------|------|----------|---------|
| trial-0 | 99 | 99 | 100 | 100 | 100 | 100 |
| trial-1 | 99 | 99 | 100 | 100 | 100 | 100 |
| trial-2 | 99 | 99 | 100 | 100 | 100 | 100 |

20 of 20 requests valid on every trial, 233 of 233 response checks passed. Quality sits at 99 rather than 100 because the single best free model (1475) has a per-trial budget of about ten calls while seventeen requests want it, so the rest take the next best (1470).

## Files

- `main.ts` — the server: the brief, the catalog, the routes, the register and the poll.
- `lib/providers.ts` — the brief's types and parser, the auth headers, the catalog probe.
- `lib/wire.ts` — the OpenAI and Anthropic client wires and their translation.
- `lib/upstream.ts` — the Gemini and Anthropic upstream wires: request builders and answer translators, in JSON and in server-sent events.

---

<sub>Entered on [make.fun](https://make.fun/?ref=atc4awx9&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
