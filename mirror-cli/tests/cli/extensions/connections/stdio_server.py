"""Hermetic stdio MCP server for host-tool integration testing."""

from __future__ import annotations

import os

import anyio
from mcp import types
from mcp.server import lowlevel
from mcp.server import stdio


async def _list_tools(
  context: object, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
  del context, params
  return types.ListToolsResult(
    tools=[
      types.Tool(
        name='echo',
        description='Echo text.',
        input_schema={
          'type': 'object',
          'properties': {'text': {'type': 'string'}},
          'required': ['text'],
        },
      )
    ]
  )


async def _call_tool(
  context: object, params: types.CallToolRequestParams
) -> types.CallToolResult:
  del context, params
  assert 'OPENROUTER_API_KEY' not in os.environ
  return types.CallToolResult(
    content=[types.TextContent(type='text', text='echoed')]
  )


async def _main() -> None:
  server = lowlevel.Server(
    'integration-test', on_list_tools=_list_tools, on_call_tool=_call_tool
  )
  async with stdio.stdio_server() as (reader, writer):
    await server.run(reader, writer, server.create_initialization_options())


anyio.run(_main)
