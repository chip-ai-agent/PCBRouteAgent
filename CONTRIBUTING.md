# Contributing

Bug reports and pull requests are welcome.

```bash
git clone --recurse-submodules https://github.com/chip-ai-agent/PCBRouteAgent.git && cd PCBRouteAgent
pip install -e ".[llm,test]"
python -m unittest discover -s route_agent/tests
```

Please run the unit tests before opening a pull request, and never commit API keys, run outputs,
routed boards or datasets.
