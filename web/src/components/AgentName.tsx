import { isBaseline, splitAgent } from "../format";

export function AgentName({ agent }: { agent: string }) {
  const [strategy, model] = splitAgent(agent);
  return (
    <span>
      <span className="agent-name">{strategy}</span>
      {model ? (
        <span className="agent-model"> · {model}</span>
      ) : (
        isBaseline(agent) && <span className="pill">baseline</span>
      )}
    </span>
  );
}
