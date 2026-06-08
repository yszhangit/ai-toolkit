# Caption / System Prompts

Drop `.txt` or `.md` files in this folder. Each file is one selectable system
prompt for the image captioners (Qwen3-VL, etc.). The **filename (without the
extension) is the title** shown in the "Load System Prompt" dropdown in the
Caption Dataset modal.

Examples of useful titles:

- `CN NL for style.txt`
- `EN tag for concept.txt`
- `CN for face lora.txt`

Selecting one fills the editable "Caption Prompt" box with the file's contents.
You can still tweak it per-job afterward — the job stores the final text, so
editing a file later won't change already-queued jobs.
