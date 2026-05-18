"""Command-line interface (Section 17)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from .agent import run_task
from .litellm_client import LiteLLMClient
from .notebook_exec import execute_notebook
from .skills import SkillRepository
from .task_graph import TaskGraph
from .transform import builtin_skills_root

app = typer.Typer(help="Notebook-native budget-aware agent (Retrieve → Compose → Transform → Generate)")
console = Console()


def _load_budget(path: Path | None) -> dict | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Budget file must be a JSON object, got {type(data).__name__}")
    return data


def _load_parameters(path: Path | None) -> dict:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Parameters file must be a JSON object, got {type(data).__name__}")
    return data


def _resolve_llm(use_llm: bool, fake_response: str | None) -> LiteLLMClient | None:
    if not use_llm and fake_response is None:
        return None
    if fake_response is not None:
        return LiteLLMClient(provider="fake", fake_response=fake_response)
    return LiteLLMClient()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Project directory to initialize."),
) -> None:
    """Initialize a notebook-agent project (creates skills/ and runs/ dirs)."""
    directory = Path(directory).resolve()
    skills_dir = directory / "skills"
    runs_dir = directory / "runs"
    skills_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    # Drop a README pointing at the built-in echo skill.
    readme = directory / "AGENT.md"
    if not readme.exists():
        readme.write_text(
            "# notebook-agent project\n\n"
            "- `skills/` — your local skill repository\n"
            "- `runs/` — task runs land here (YYYY/MM/DD/HHMMSS-slug)\n\n"
            "Built-in skills (e.g. `core.echo`) are bundled with the package.\n"
            "Try: `notebook-agent run \"Echo hello world\" --param-message hello`\n",
            encoding="utf-8",
        )
    console.print(f"[green]Initialized notebook-agent project at[/green] {directory}")


@app.command(name="run")
def run_command(
    request: str = typer.Argument(..., help="Natural-language task request."),
    budget: Path | None = typer.Option(None, "--budget", "-b", help="Path to a JSON budget file."),
    parameters: Path | None = typer.Option(None, "--params", "-p", help="Path to a JSON parameters file."),
    param: list[str] | None = typer.Option(None, "--param", help="Individual parameter as KEY=VALUE (repeatable, merges with --params)."),
    runs_root: Path = typer.Option(Path("runs"), "--runs-root", help="Root directory for run outputs."),
    skills_dir: list[Path] | None = typer.Option(None, "--skills-dir", "-s", help="Extra skill directory (repeatable)."),
    title: str | None = typer.Option(None, "--title", help="Override task title."),
    use_llm: bool = typer.Option(False, "--llm/--no-llm", help="Enable LiteLLM (LM Studio by default)."),
    fake_llm_response: str | None = typer.Option(None, "--fake-llm", help="Use the fake LLM provider with this response (for testing)."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of rich output."),
) -> None:
    """Run a task end-to-end."""
    budget_dict = _load_budget(budget)
    params = _load_parameters(parameters)
    for kv in param or []:
        if "=" not in kv:
            raise typer.BadParameter(f"--param must be KEY=VALUE, got {kv!r}")
        k, _, v = kv.partition("=")
        params[k.strip()] = v
    llm = _resolve_llm(use_llm, fake_llm_response)
    result = run_task(
        request,
        runs_root=runs_root,
        parameters=params,
        budget=budget_dict,
        title=title,
        skill_dirs=list(skills_dir or []),
        llm=llm,
    )
    if json_out:
        typer.echo(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        color = "green" if result.success else "red"
        console.print(f"[{color}]Status:[/{color}] {result.task.status}")
        console.print(f"[bold]Stage used:[/bold] {result.stage_used}")
        console.print(f"[bold]Directory:[/bold] {result.task.directory}")
        if result.extras.get("parameter_extractor_error"):
            console.print(
                f"[yellow]Warning:[/yellow] parameter inference failed: "
                f"{result.extras['parameter_extractor_error']}"
            )
        if result.answer:
            console.print("[bold]Answer:[/bold]")
            console.print(result.answer)
    raise typer.Exit(code=0 if result.success else 1)


@app.command(name="search-skills")
def search_skills_command(
    query: str = typer.Argument(..., help="Lexical search query."),
    skills_dir: list[Path] | None = typer.Option(None, "--skills-dir", "-s", help="Extra skill directory (repeatable)."),
    top: int = typer.Option(10, "--top", help="Maximum number of results."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of rich output."),
) -> None:
    """Search built-in + project skills."""
    roots: list[Path | str] = [builtin_skills_root()]
    if skills_dir:
        roots.extend(skills_dir)
    cwd_skills = Path.cwd() / "skills"
    if cwd_skills.exists():
        roots.append(cwd_skills)
    repo = SkillRepository(roots)
    results = repo.search(query, top_k=top)
    if json_out:
        typer.echo(json.dumps([r.to_dict() for r in results], indent=2))
        return
    if not results:
        console.print("[yellow]No matches[/yellow]")
        return
    table = Table(title=f"Skill search: {query!r}")
    table.add_column("Score", justify="right")
    table.add_column("Skill ID")
    table.add_column("Name")
    table.add_column("Tags")
    for r in results:
        table.add_row(f"{r.score:.2f}", r.skill.skill_id, r.skill.name, ", ".join(r.skill.tags))
    console.print(table)


@app.command(name="execute-notebook")
def execute_notebook_command(
    notebook: Path = typer.Argument(..., help="Path to the notebook to execute."),
    params: Path | None = typer.Option(None, "--params", help="JSON parameters file."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output notebook path."),
    run_dir: Path | None = typer.Option(None, "--run-dir", help="Run directory for result.json and logs."),
) -> None:
    """Execute a parameterized notebook with Papermill."""
    parameters_dict = _load_parameters(params)
    res = execute_notebook(
        notebook,
        parameters=parameters_dict,
        output_path=output,
        run_dir=run_dir,
    )
    console.print(f"success={res.success}")
    if res.result is not None:
        console.print_json(json.dumps(res.result))
    raise typer.Exit(code=0 if res.success else 1)


def _render_tree(graph: TaskGraph) -> Tree:
    def _add(node, task):
        for child in task.children:
            sub = node.add(f"[cyan]{child.directory.name}[/cyan] — {child.title} ({child.status})")
            _add(sub, child)

    root_label = f"[bold]{graph.root.directory.name}[/bold] — {graph.root.title} ({graph.root.status})"
    tree = Tree(root_label)
    _add(tree, graph.root)
    return tree


@app.command(name="graph")
def graph_command(
    directory: Path = typer.Argument(..., help="Task directory (root of the task graph)."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of tree."),
) -> None:
    """Render the task graph rooted at DIRECTORY."""
    graph = TaskGraph.load(directory)
    if json_out:
        typer.echo(json.dumps(graph.to_dict(), indent=2))
    else:
        console.print(_render_tree(graph))


@app.command(name="manifest")
def manifest_command(
    directory: Path = typer.Argument(..., help="Task directory."),
) -> None:
    """Print the manifest.json for a task."""
    p = Path(directory) / "manifest.json"
    if not p.exists():
        console.print(f"[red]No manifest.json at {directory}[/red]")
        raise typer.Exit(code=1)
    typer.echo(p.read_text(encoding="utf-8"))


@app.command(name="mcp")
def mcp_command(
    transport: str = typer.Option("stdio", "--transport", help="MCP transport (stdio for now)."),
    skills_dir: list[Path] | None = typer.Option(None, "--skills-dir", "-s"),
    runs_root: Path = typer.Option(Path("runs"), "--runs-root"),
) -> None:
    """Start the MCP server exposing notebook-agent tools."""
    from .mcp_server import serve_stdio

    if transport != "stdio":
        console.print(f"[red]Unsupported transport: {transport}[/red]")
        raise typer.Exit(1)
    serve_stdio(runs_root=runs_root, skill_dirs=list(skills_dir or []))


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
