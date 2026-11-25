# \!/usr/bin/env python3
"""
Test the researcher locally before deployment
"""

import asyncio
from context import get_agent_instructions, DEFAULT_RESEARCH_PROMPT
from tools import ingest_financial_document
from finalex_runtime import Agent, Runner, LitellmModel
from dotenv import load_dotenv

load_dotenv(override=True)


async def test_local():
    """Test the researcher agent locally."""
    print("Testing researcher agent locally...")
    print("=" * 60)

    # Test with no topic (agent picks)
    query = DEFAULT_RESEARCH_PROMPT

    try:
        agent = Agent(
            name="Finalex Investment Researcher",
            instructions=get_agent_instructions(),
            model=LitellmModel("gpt-4.1-mini"),
        )
        result = await Runner.run(agent, input=query)
        ingest_financial_document("local research", result.final_output)

        print("\nRESULT:")
        print("=" * 60)
        print(result.final_output)
        print("=" * 60)
        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_local())
