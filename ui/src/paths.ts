import path from 'path';
export const TOOLKIT_ROOT = path.resolve('@', '..', '..');
export const defaultTrainFolder = path.join(TOOLKIT_ROOT, 'output');
export const defaultDatasetsFolder = path.join(TOOLKIT_ROOT, 'datasets');
export const defaultDataRoot = path.join(TOOLKIT_ROOT, 'data');
// User-authored caption/system prompts. Each .txt/.md file's name (sans
// extension) is the title shown in the captioner's "Load System Prompt" dropdown.
export const captionPromptsFolder = path.join(TOOLKIT_ROOT, 'caption_prompts');
