# rust-ai-agents-wasm

WebAssembly bindings for running AI agents directly in the browser.

## Features

- **Browser-native**: Run AI agents entirely in the browser with no server required
- **Multiple Providers**: Support for OpenAI, Anthropic, and OpenRouter
- **Streaming**: Real-time streaming responses
- **TypeScript**: Full TypeScript type definitions included
- **Lightweight**: Optimized WASM bundle

## Installation

```bash
npm install rust-ai-agents-wasm
# or
yarn add rust-ai-agents-wasm
# or
pnpm add rust-ai-agents-wasm
```

## Quick Start

### Vanilla JavaScript

```javascript
import init, { WasmAgent, WasmAgentConfig } from 'rust-ai-agents-wasm';

// Initialize the WASM module
await init();

// Create an agent
const config = new WasmAgentConfig('openai', 'sk-...', 'gpt-4o-mini');
config.system_prompt = 'You are a helpful assistant.';
config.temperature = 0.7;

const agent = new WasmAgent(config);

// Chat (non-streaming)
const response = await agent.chat('Hello!');
console.log(response.content);

// Chat (streaming)
const stream = await agent.chatStream('Tell me a story');
const reader = stream.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value, { stream: true });
  // Parse SSE data...
}

// Clear history
agent.clear();

// Free memory when done
agent.free();
```

### React

```tsx
import { useWasmAgent } from 'rust-ai-agents-wasm/react';

function Chat() {
  const { sendMessageStream, messages, isStreaming, isReady } = useWasmAgent({
    provider: 'openai',
    apiKey: process.env.REACT_APP_OPENAI_API_KEY!,
    model: 'gpt-4o-mini',
    systemPrompt: 'You are a helpful assistant.',
  });

  return (
    <div>
      {messages.map((msg, i) => (
        <div key={i} className={msg.role}>
          {msg.content}
        </div>
      ))}
      <button 
        onClick={() => sendMessageStream('Hello!')}
        disabled={!isReady || isStreaming}
      >
        Send
      </button>
    </div>
  );
}
```

## API Reference

### WasmAgentConfig

```typescript
const config = new WasmAgentConfig(
  provider: string,  // 'openai' | 'anthropic' | 'openrouter'
  apiKey: string,
  model: string
);

// Optional settings
config.system_prompt = 'You are...';
config.temperature = 0.7;      // 0.0 - 2.0
config.max_tokens = 4096;
```

### WasmAgent

```typescript
const agent = new WasmAgent(config);

// Non-streaming chat
const response = await agent.chat(message: string);
// Returns: { content: string, tool_calls?: [...], usage?: {...} }

// Streaming chat
const stream = await agent.chatStream(message: string);
// Returns: ReadableStream

// Get conversation history
const history = agent.getHistory();
// Returns: Array<{ role: string, content: string }>

// Clear conversation
agent.clear();

// Free memory
agent.free();
```

### WasmMessage

```typescript
const userMsg = WasmMessage.user('Hello');
const assistantMsg = WasmMessage.assistant('Hi there!');

console.log(userMsg.role);     // 'user'
console.log(userMsg.content);  // 'Hello'
```

## Supported Providers

| Provider | Models | Streaming | Notes |
|----------|--------|-----------|-------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc. | ✅ | |
| Anthropic | claude-3-opus, claude-3-sonnet, etc. | ✅ | Requires `anthropic-dangerous-direct-browser-access` header |
| OpenRouter | 100+ models | ✅ | Best for accessing multiple providers |

## Security Considerations

⚠️ **API Keys in Browser**: This library is designed for prototyping and demos. In production:

1. **Never expose API keys** in client-side code
2. Use a backend proxy to handle API calls
3. Implement proper authentication and rate limiting

For production use, consider:
- Using OpenRouter with a limited-scope API key
- Implementing a server-side proxy
- Using Anthropic's `anthropic-dangerous-direct-browser-access` header only for demos

## Browser Compatibility

| Browser | Supported |
|---------|-----------|
| Chrome 89+ | ✅ |
| Firefox 89+ | ✅ |
| Safari 15+ | ✅ |
| Edge 89+ | ✅ |

Requires:
- WebAssembly
- Fetch API
- ReadableStream
- TextDecoder

## Examples

See the `/examples` directory for:
- `browser/index.html` - Vanilla JavaScript demo
- `react/` - React hook and component examples

## Development

```bash
# Build WASM package
cd crates/wasm
wasm-pack build --target web --out-dir pkg

# Run tests
wasm-pack test --headless --firefox
```

## License

Apache-2.0
