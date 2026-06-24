import sys

import click
from dotenv import load_dotenv

from deep_research.coordinator import run_deep_research

load_dotenv()


def _log(message: str) -> None:
    click.echo(message, err=True)


@click.command()
@click.argument("query", required=False)
@click.option("-o", "--output", default="results.md", help="Output file path.")
def main(query: str | None, output: str) -> None:
    """Run a deep research query and write the report to a file."""
    if not query:
        query = click.prompt("Enter your research query")

    click.echo(f"Query: {query}", err=True)
    click.echo("", err=True)

    result = run_deep_research(query=query, log=_log)

    with open(output, "w") as f:
        f.write(result)

    click.echo(f"\nReport saved to {output}", err=True)


if __name__ == "__main__":
    main()
