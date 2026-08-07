# Example run

This is a real, unedited transcript -- not a mockup. It was produced by
actually running:

```
$ mkdir /tmp/demo_ws && cat > /tmp/demo_ws/calc.py << 'EOF'
def add(a, b):
    return a - b  # bug: should be addition


def multiply(a, b):
    return a * b
EOF

$ codeagent run "Find and fix the bug in calc.py" \
    --workdir /tmp/demo_ws \
    --provider mock \
    --script examples/demo_actions.json
```

`examples/demo_actions.json` scripts a `MockLLM` with the five actions a
real model might plausibly choose for this task, in order -- so the whole
loop runs deterministically with no API key or network access. Swap
`--provider mock --script ...` for `--provider openai` or `--provider
anthropic` to have a real model choose those actions live instead; the
agent loop, tools, and safety policy underneath are identical either way.

## Output

```
[step 0] thought: Let's see what's in the workspace first.
  action: list_dir({'path': '.'})
  observation (ok): f calc.py
[step 1] thought: Read calc.py to find the bug.
  action: read_file({'path': 'calc.py'})
  observation (ok): def add(a, b):
    return a - b  # bug: should be addition


def multiply(a, b):
    return a * b

[step 2] thought: add() is doing subtraction instead of addition. Fix it.
  action: edit_file({'path': 'calc.py', 'old': 'return a - b  # bug: should be addition', 'new': 'return a + b'})
  observation (ok): edited calc.py
[step 3] thought: Verify the fix by running a quick sanity check.
  action: run_shell({'command': 'python3 -c "import calc; assert calc.add(2, 3) == 5; print(\'add(2, 3) =\', calc.add(2, 3))"'})
  observation (ok): add(2, 3) = 5

[exit code 0]
[step 4] thought: Fix confirmed working, task complete.
  action: finish({'summary': 'Fixed add() in calc.py: it was subtracting instead of adding. Verified add(2, 3) now returns 5.'})
  observation (ok): Fixed add() in calc.py: it was subtracting instead of adding. Verified add(2, 3) now returns 5.

SUCCESS after 5 step(s)
Fixed add() in calc.py: it was subtracting instead of adding. Verified add(2, 3) now returns 5.
```

`calc.py` on disk afterward:

```python
def add(a, b):
    return a + b


def multiply(a, b):
    return a * b
```

Notice step 2 (`edit_file`) required the *exact* pre-existing line,
including the misleading `# bug: should be addition` comment -- that's
the uniqueness discipline described in the README: `edit_file` fails
loudly rather than guessing if `old` isn't found verbatim, or isn't
unique in the file.
