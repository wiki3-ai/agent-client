# Focused Spec for a Notebook-Centric Coding Agent

## What the sources say the core should optimize for

Papermill is a very good fit for the launch mechanism you described because its core model is a **parameterized notebook**: a cell tagged parameters provides defaults, Papermill injects an injected-parameters cell with overrides, and the notebook is then executed to an output notebook using the Python API execute\_notebook(input, output, parameters=...). Papermill’s execution machinery also updates notebook and cell metadata during execution and saves the notebook state to the output path, which is exactly the behavior you want for notebook-native provenance. One important implementation detail is that interdependent derived values should not live in the parameters cell if you expect runtime overrides, because Papermill only injects overridden values afterward. [\[1\]](https://papermill.readthedocs.io/en/latest/usage-parameterize.html)

The Jupyter notebook format itself can carry the persistence model you want. A notebook is a JSON document with top-level metadata and cells; metadata can hold arbitrary JSON-like information at the notebook, cell, and output levels; and rich outputs are stored as MIME bundles keyed by MIME type, including custom application/vnd...+json types. The format also supports cell tags, collapsed or hidden outputs, and source/output visibility flags, which makes it suitable for storing visible plan/result cells and less-prominent provenance or machine-readable state in metadata or hidden cells instead of sidecar JSON files. [\[2\]](https://nbformat.readthedocs.io/en/latest/format_description.html)

For the notebook UX, IPython supports exactly the two user-entry paths you asked for: a normal Python function and Jupyter magics. IPython documents line magics, cell magics, and combined line/cell magics, and it also documents extensions loaded through load\_ipython\_extension(ipython), where the extension can register custom magics at startup or through %load\_ext. At the same time, IPython notes that magics are kernel-specific, which is a useful reminder to keep the first implementation as an IPython/Jupyter extension while leaving room for a later dedicated kernel package. [\[3\]](https://ipython.readthedocs.io/en/latest/config/custommagics.html)

Your reference projects point in the same direction. The BabyAGI notebook you linked uses a notebook flow with a task-creation chain, a task-prioritization chain, and an execution chain, and it exposes three especially relevant built-in capabilities: Search, a TODO planner, and a Python REPL/code-execution tool. But that notebook also prompts users for provider or search API keys inside cells, which is the opposite of the low-friction run\_task(prompt) UX you want now. Meanwhile, the agent-client-kernel project shows a Jupyter-kernel-first user experience in which users interact directly in notebook cells and configure behavior through a magic command, and the Ontological Engineer materials explicitly emphasize notebook-based provenance, notebook-based outputs, human feedback, and optional MLflow tracing rather than making MLflow a hard requirement. [\[4\]](https://raw.githubusercontent.com/jimwhite/babyagi-langchain/refs/heads/main/baby_agi_with_programming_agent.ipynb)

## Recommended scope for the first real delivery

The first delivery should be **much narrower than a general autonomous agent framework**. It should ship the smallest thing that proves the notebook-centric interaction model works end to end: a root Papermill template notebook that accepts a goal, runs an agent loop for a bounded number of LLM turns, records its planning and execution inside the notebook, and returns a user-visible response that can then be continued through the same notebook session. Papermill’s Python API is sufficient for this; there is no need to expose a user CLI at all. [\[5\]](https://papermill.readthedocs.io/en/latest/usage-execute.html)

The initial persistence model should be **notebook-only**. The executed notebook should hold visible markdown/code/output cells for the ordinary human reading path, while notebook metadata, cell metadata, and MIME-bundled outputs should hold machine-readable state such as conversation ID, parent/root notebook references, turn counter, last plan, TODO list, retrieved artifacts, generated notebook paths, execution status, and hidden diagnostics. That matches both the notebook file format and your requirement to avoid sidecar JSON manifests. [\[6\]](https://nbformat.readthedocs.io/en/latest/format_description.html)

The initial budget model should be intentionally simple: **the only enforced quota is the number of autonomous LLM turns allowed before the agent must report back to the user**. Do not start with token, cost, or wall-time budgets as hard gates for user-facing execution. Those can be logged for later profiling, but the enforced control loop should just be “how many turns may the agent take without surfacing progress or asking for continuation.” This will prevent the premature failure mode you observed with tiny artificial token budgets while still creating a clean user checkpoint model.

The minimal built-in capability set should mirror the useful parts of the BabyAGI pattern, but in your notebook-native architecture: local search/retrieval, optional web search, TODO planning, code/notebook generation, and notebook execution. DSPy is a good fit for compact planning and module composition because it is designed around signatures, modules, and optimizers for modular AI software, and DSPy can target local or OpenAI-compatible endpoints. LiteLLM is a reasonable plumbing layer because it supports OpenAI-compatible endpoints and also has direct LM Studio support; LM Studio itself exposes OpenAI-compatible endpoints at a configurable base URL, with the docs using http://localhost:1234/v1 in examples. [\[7\]](https://dspy.ai/)

## Focused spec prompt for the coding agent

The prompt below is the version I would hand to the coding agent. It is intentionally short, opinionated, and scoped to the core functionality rather than the full long-term architecture.

Build the core MVP of a notebook-centric coding agent.

Primary goal:  
Create a Jupyter-first coding agent whose main user API is:

    run\_task(prompt: str) \-\> object

and which also installs equivalent Jupyter magics:

    %task \<prompt\>  
    %%task  
    \<multi-line prompt body\>

This is NOT a user CLI project. Do not build a user-facing CLI. Internal tools may use subprocess or shell execution when necessary, but the user experience is notebook-native.

Core execution model:  
\- The agent is launched by executing a Papermill-parameterized notebook template.  
\- The root notebook template must contain a Papermill parameters cell.  
\- One required parameter is the user goal/prompt.  
\- Other parameters are configuration defaults such as provider/model/base URL/max autonomous turns.  
\- Use Papermill programmatically from Python to execute notebooks.

State and persistence:  
\- Canonical persistence is in notebooks, not sidecar JSON/YAML/JSONL files.  
\- Store ordinary human-visible information in notebook cells.  
\- Store machine-readable state, provenance, hidden diagnostics, and structured result records in notebook metadata, cell metadata, and MIME-bundled outputs.  
\- The executed notebook is the durable task record.

Top-level UX requirements:  
\- A notebook user should be able to do only this:  
      run\_task("your request here")  
  or:  
      %task your request here  
  or:  
      %%task  
      your multi-line request here  
\- The normal user must NOT be required to manually configure an LLM object, pass a registry, or pass a large parameter block.

Default configuration:  
\- Provide sensible defaults for finding/configuring the LLM provider and model.  
\- Prefer environment-variable and config-based defaults.  
\- Support local OpenAI-compatible endpoints and LM Studio via LiteLLM.  
\- Keep the configuration easy to override, but hidden by default for normal users.

Agent behavior for the MVP:  
\- Accept the user prompt.  
\- Create or reuse a root task notebook execution.  
\- Make a visible short plan/TODO list in the notebook.  
\- Search locally for an existing skill/notebook/code path that can satisfy the task.  
\- If no suitable skill exists, generate a new executable notebook or notebook cells to perform the task.  
\- Execute the resulting notebook with Papermill.  
\- Report the result back to the user in the notebook.  
\- Preserve enough state to continue the interaction after the response.

Continuation model:  
\- After the first response, the user must be able to continue the same task thread with feedback, corrections, “continue”, or “not satisfactory”.  
\- The conversation/task state must stay attached to the notebook record.  
\- The only enforced budget limit in this first version is:  
      max\_autonomous\_turns  
  meaning the maximum number of LLM turns the agent may take before it must report back to the user.  
\- If the agent reaches this limit and believes it can still make progress, it should say so and allow the user to continue.

Built-in core capabilities required in MVP:  
\- local search / retrieve  
\- optional web search  
\- TODO planning  
\- notebook/code generation  
\- execute notebook  
\- basic continuation / feedback handling

Architecture constraints:  
\- Use Python.  
\- Use Papermill for notebook execution.  
\- Use LiteLLM for model access.  
\- Use DSPy where useful for planning or modular LM calls, but do not over-engineer the first delivery.  
\- Do not require MLflow or any external observability stack for the core workflow.  
\- Design so later promotion to a dedicated Jupyter kernel is straightforward.

Implementation priorities:  
\- Working end-to-end behavior is more important than a large framework.  
\- The agent must be able to perform a task that does not already have a prebuilt skill.  
\- “Echo the prompt” is NOT acceptable except as a tiny smoke test.  
\- The first real acceptance path must prove planning plus generation plus execution.

Required acceptance tests:  
1\. Smoke test:  
   \- run\_task("say hello") returns a notebook-visible response.

2\. Notebook magic test:  
   \- %task and %%task call the same underlying request path as run\_task(prompt).

3\. Real generation test:  
   \- With no pre-existing word-count skill, run:  
         run\_task("Create and execute a notebook that counts the words in: hello from the graph notebook agent")  
   \- The agent must:  
         plan,  
         notice no suitable existing skill,  
         generate the needed notebook/code,  
         execute it,  
         and return the correct result (6).

4\. Continuation test:  
   \- After the previous task, the user can say “that result is wrong, try again” or “continue”.  
   \- The agent resumes the same task state rather than starting from scratch.

5\. Turn budget test:  
   \- Configure max\_autonomous\_turns to a low value.  
   \- Verify the agent stops after hitting the turn limit, reports progress/status, and can continue after the next user message.

6\. Papermill integration test:  
   \- Execute the root parameterized notebook and confirm the output notebook contains result/provenance/state.

7\. Notebook persistence test:  
   \- Verify that task state is present in notebook metadata/cell metadata/MIME outputs and does not depend on sidecar JSON files.

Testing requirements:  
\- Write unit tests for core modules.  
\- Write integration tests that execute real notebooks with Papermill.  
\- Write live acceptance tests that use a real configured LLM endpoint for at least the main happy path.  
\- Do not mark the milestone complete until the live acceptance tests pass.

Deliverables:  
\- Python package implementing run\_task(prompt)  
\- IPython extension registering %task and %%task  
\- root Papermill notebook template  
\- minimal built-in skills/modules for retrieval, planning, generation, execution, continuation  
\- automated test suite including live integration tests  
\- short notebook demo showing the intended user UX

This prompt centers the build on one **root parameterized notebook**, one **simple user entrypoint**, one **turn-based autonomy budget**, and one **notebook-native persistence model**, because those are the strongest overlaps among Papermill’s execution model, IPython’s extension/magic model, the notebook file format, the BabyAGI-inspired planning pattern, and your existing notebook-centric downstream work. [\[8\]](https://papermill.readthedocs.io/en/latest/usage-parameterize.html)

## Acceptance criteria that matter more than the code structure

The most important acceptance criterion is **non-trivial task completion without a pre-authored skill**. The BabyAGI notebook is useful inspiration because it already combines Search, TODO planning, and a code-execution tool, but your MVP should improve on it by removing the manual provider-key prompts and by making the root workflow a reusable Papermill notebook that can generate and execute task notebooks rather than just echo or route text. [\[9\]](https://raw.githubusercontent.com/jimwhite/babyagi-langchain/refs/heads/main/baby_agi_with_programming_agent.ipynb)

The second critical criterion is **continuable notebook interaction**. The agent-client-kernel README shows a user experience where notebook cells are the interaction surface and magic commands are used for session and configuration management. Your MVP does not need to become a full custom kernel yet, but it should already feel like notebook-native software: run\_task(prompt), %task, and %%task should all operate on the same underlying task/session state. [\[10\]](https://github.com/wiki3-ai/agent-client-kernel)

The third criterion is **notebook-native provenance**. The Ontological Engineer design documents explicitly keep notebook-based provenance and notebook-based outputs, including CID-linked incremental processing and human feedback tied to content rather than transient positions. That makes it a good downstream benchmark: the core agent should not force MLflow or a separate trace store, but it should make later observability integration easy by keeping the core record in notebooks. [\[11\]](https://raw.githubusercontent.com/wiki3-ai/ontological-engineer/main/DSPY_PIPELINE_DESIGN.md)

The fourth criterion is **real live integration testing**. Because Papermill is supposed to execute real notebooks and because the user entrypoint is a Jupyter/IPython experience, the acceptance suite should actually execute notebooks, load the IPython extension, invoke %task and %%task, run against a real model endpoint, and verify notebook metadata and outputs. A pure unit-test pass is not enough for this milestone. [\[12\]](https://papermill.readthedocs.io/en/latest/usage-execute.html)

## Failure modes to explicitly forbid in the spec

Do not allow the coding agent to satisfy the milestone with an **echo skill**, a fake planner, a placeholder “future CLI,” or a wrapper that requires the notebook user to manually construct and pass an LLM object. The BabyAGI notebook’s manual entry of OpenAI and search credentials is a good example of the kind of friction you are trying to eliminate in the core UX. [\[9\]](https://raw.githubusercontent.com/jimwhite/babyagi-langchain/refs/heads/main/baby_agi_with_programming_agent.ipynb)

Do not allow persistence to drift into sidecar files just because they are easier to debug. The notebook format already supports arbitrary metadata, output metadata, custom MIME bundles, tags, and hidden/collapsed display controls, so the spec should explicitly force the coding agent to use those notebook-native mechanisms first. [\[6\]](https://nbformat.readthedocs.io/en/latest/format_description.html)

Do not make MLflow, LangSmith, or any other external tracing stack mandatory in the core path. The Ontological Engineer design uses MLflow for tracing and human feedback, but it also clearly treats notebook outputs and notebook provenance as first-class. For your MVP, observability should be optional and additive, not part of the launch path. [\[11\]](https://raw.githubusercontent.com/wiki3-ai/ontological-engineer/main/DSPY_PIPELINE_DESIGN.md)

Do not overcomplicate the first model layer. DSPy is valuable because it encourages modular signatures and programmable LM components, and LiteLLM plus LM Studio give you a practical path to local or OpenAI-compatible models. But the first milestone should use those libraries to simplify the agent, not to introduce a large optimization stack before the notebook UX works. [\[7\]](https://dspy.ai/)

---

[\[1\]](https://papermill.readthedocs.io/en/latest/usage-parameterize.html) [\[8\]](https://papermill.readthedocs.io/en/latest/usage-parameterize.html) Parameterize \- papermill 2.7.0 documentation

[https://papermill.readthedocs.io/en/latest/usage-parameterize.html](https://papermill.readthedocs.io/en/latest/usage-parameterize.html)

[\[2\]](https://nbformat.readthedocs.io/en/latest/format_description.html) [\[6\]](https://nbformat.readthedocs.io/en/latest/format_description.html) The Notebook file format — nbformat 5.10 documentation

[https://nbformat.readthedocs.io/en/latest/format\_description.html](https://nbformat.readthedocs.io/en/latest/format_description.html)

[\[3\]](https://ipython.readthedocs.io/en/latest/config/custommagics.html) Defining custom magics — IPython 9.14.0.dev documentation

[https://ipython.readthedocs.io/en/latest/config/custommagics.html](https://ipython.readthedocs.io/en/latest/config/custommagics.html)

[\[4\]](https://raw.githubusercontent.com/jimwhite/babyagi-langchain/refs/heads/main/baby_agi_with_programming_agent.ipynb) [\[9\]](https://raw.githubusercontent.com/jimwhite/babyagi-langchain/refs/heads/main/baby_agi_with_programming_agent.ipynb) raw.githubusercontent.com

[https://raw.githubusercontent.com/jimwhite/babyagi-langchain/refs/heads/main/baby\_agi\_with\_programming\_agent.ipynb](https://raw.githubusercontent.com/jimwhite/babyagi-langchain/refs/heads/main/baby_agi_with_programming_agent.ipynb)

[\[5\]](https://papermill.readthedocs.io/en/latest/usage-execute.html) [\[12\]](https://papermill.readthedocs.io/en/latest/usage-execute.html) Execute \- papermill 2.7.0 documentation

[https://papermill.readthedocs.io/en/latest/usage-execute.html](https://papermill.readthedocs.io/en/latest/usage-execute.html)

[\[7\]](https://dspy.ai/) DSPy

[https://dspy.ai/](https://dspy.ai/)

[\[10\]](https://github.com/wiki3-ai/agent-client-kernel) GitHub \- wiki3-ai/agent-client-kernel: A Zed Agent Client Protocol (ACP) Jupyter Kernel · GitHub

[https://github.com/wiki3-ai/agent-client-kernel](https://github.com/wiki3-ai/agent-client-kernel)

[\[11\]](https://raw.githubusercontent.com/wiki3-ai/ontological-engineer/main/DSPY_PIPELINE_DESIGN.md) raw.githubusercontent.com

[https://raw.githubusercontent.com/wiki3-ai/ontological-engineer/main/DSPY\_PIPELINE\_DESIGN.md](https://raw.githubusercontent.com/wiki3-ai/ontological-engineer/main/DSPY_PIPELINE_DESIGN.md)