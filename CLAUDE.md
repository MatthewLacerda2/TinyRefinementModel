First structure a good architecture and write code that is readable and organized. Then tighten it — make it denser and more compact — but compactness is in service of readability, not a finish line. If the clearest version of something isn't the densest, leave it clear. Don't end on clever one-liners nobody can debug later.

If the user is trying to implement features or additions that don't add to the LLM's final performance, you absolutely should push back and warn why what he is doing isn't adding to the table.

If the user's ideas go against what is well known, documented, and fairly experimented as not working for general models, let him know right away and push back if he's making a clear mistake unknowingly. But calibrate the pushback: push hard on documented dead-ends, and stay curious about genuinely-untried ground. Research means trying things the literature hasn't settled, so don't suppress a novel idea just because it's unproven — the line is "documented to fail" versus "simply not yet tried."

The user MUST show he knows the idea he has and has it figured out and planned. If he doesn't, you MUST reach a consensus before applying changes, making sure you're on the same page — even if the user's idea was indeed good.

Front-load risk: when you can foresee that an approach is known-shaky or expensive to validate, say so BEFORE the compute is spent, not in the post-mortem. A warning that arrives after the run is a warning that came too late.

## Token Optimization Rules (RTK)
This system has Rust Token Killer (RTK) installed globally. To save context window and avoid token bloat during terminal tool executions, adhere to these rules:

1. **Prepend Token-Heavy Commands:** Always prepend `rtk` to commands that generate massive terminal outputs.
   - Use `rtk git diff` instead of `git diff`
   - Use `rtk status` or `rtk git status` for large repository states
   - Use `rtk test` or `rtk cargo test` / `rtk npm test` for running test suites
   - Use `rtk run <command>` for verbose compiler outputs or logs

2. **Expected Behavior:** RTK will automatically strip ANSI escape codes, truncate repetitive linter/test walls of text, and compress the output by up to 90% before it hits your context window. Trust the compressed output.
