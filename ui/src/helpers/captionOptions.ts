import { GroupedSelectOption, SelectOption } from "@/types";

type CaptionGroup = 'image' | 'music';
type AdditionalSections = 'caption.model_name_or_path2' | 'caption.caption_prompt' | 'caption.max_res' | 'caption.max_new_tokens' | 'caption.gen_params' | 'caption.fixed_caption';

// Qwen3-VL sampling preset, loaded into the job when the model is selected.
// These are user-tunable starting points (seeded from Qwen3-VL-Instruct's
// https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
// generation_config defaults). Edit the values here to change what pre-fills a
// new Qwen3-VL caption job. Any field can still be overridden per-job in the UI.
export const qwen3vlPreset = {
    max_res: 1024,
    max_new_tokens: 2048,
    temperature: 0.2,
    top_p: 0.8,
    top_k: 20,
    repetition_penalty: 1.05,
};

// Ideogram4 emits structured JSON, so it wants a lower, more deterministic
// sampling preset than the free-form Qwen3-VL captioner. Same edit-here pattern.
export const ideogram4Preset = {
    temperature: 0.1,
    top_p: 0.7,
    top_k: 20,
    repetition_penalty: 1.05,
};

export interface CaptionOption {
    name: string;
    label: string;
    group: CaptionGroup;
    hasMultiLinePrompts?: boolean;
    minNewTokens?: number;
    defaults?: { [key: string]: any };
    additionalSections?: AdditionalSections[];
    name_or_path_options?: SelectOption[];
    name_or_path2_options?: SelectOption[];
}

const defaultNameOrPath = '';

const extensionsAudio = ['mp3', 'wav', 'flac', 'ogg'];
const extensionsImage = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];

const defaultExtensions = [...extensionsImage];

const defaultImageCaptionPrompt = "Caption this image as if you were going to try to generate it with an image generator. Be thurough and describe everything in the image. Be decisive by stating things as they are. Do not say things like \"It appears that\" Or \"possibly\". Start out with things like \"A person on the beach\" or \"A black dragon\". No preamble. Just get to the point.";

// Editable ADDITIONAL INSTRUCTIONS block injected into the Ideogram system prompt.
// Users can tweak this for dataset-specific guidance without altering the fixed
// output contract, element/background rules, or bbox format.
const defaultIdeogramCaptionPrompt = "Describe only what is actually visible in the image — never invent, add, or infer content that is not present. Identify the medium (photograph, illustration, 3D render, or graphic design) and name any recognizable people, brands, characters, or landmarks. Be decisive and specific: commit to one concrete value per attribute (one color, one material, one pose) and avoid hedging. Transcribe every piece of legible text verbatim.";

export const captionerTypes: CaptionOption[] = [
    {
        name: 'AceStepCaptioner',
        label: 'Ace Step',
        group: 'music',
        defaults: {
            'config.process[0].caption.model_name_or_path': ['ACE-Step/acestep-transcriber', defaultNameOrPath],
            'config.process[0].caption.model_name_or_path2': ['ACE-Step/acestep-captioner', undefined],
            'config.process[0].caption.extensions': [extensionsAudio, defaultExtensions],
        },
        name_or_path_options: [
            { value: 'ACE-Step/acestep-transcriber', label: 'ACE-Step/acestep-transcriber' },
        ],
        name_or_path2_options: [
            { value: 'ACE-Step/acestep-captioner', label: 'ACE-Step/acestep-captioner' },
        ],
        additionalSections: [
            'caption.model_name_or_path2',
            'caption.fixed_caption',
        ],
    },
    {
        name: 'Qwen3VLCaptioner',
        label: 'Qwen3-VL',
        group: 'image',
        defaults: {
            'config.process[0].caption.model_name_or_path': ['Qwen/Qwen3-VL-8B-Instruct', defaultNameOrPath],
            'config.process[0].caption.extensions': [extensionsImage, defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultImageCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [qwen3vlPreset.max_res, undefined],
            'config.process[0].caption.max_new_tokens': [qwen3vlPreset.max_new_tokens, undefined],
            'config.process[0].caption.temperature': [qwen3vlPreset.temperature, undefined],
            'config.process[0].caption.top_p': [qwen3vlPreset.top_p, undefined],
            'config.process[0].caption.top_k': [qwen3vlPreset.top_k, undefined],
            'config.process[0].caption.repetition_penalty': [qwen3vlPreset.repetition_penalty, undefined],
        },
        name_or_path_options: [
            { value: 'Qwen/Qwen3-VL-2B-Instruct', label: 'Qwen/Qwen3-VL-2B-Instruct' },
            { value: 'Qwen/Qwen3-VL-4B-Instruct', label: 'Qwen/Qwen3-VL-4B-Instruct' },
            { value: 'Qwen/Qwen3-VL-8B-Instruct', label: 'Qwen/Qwen3-VL-8B-Instruct' },
            { value: 'Qwen/Qwen3-VL-30B-A3B-Instruct', label: 'Qwen/Qwen3-VL-30B-A3B-Instruct' },
        ],
        additionalSections: [
            'caption.caption_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
            'caption.gen_params',
        ],
    },
    {
        name: 'Ideogram4Captioner',
        label: 'Ideogram 4 Captioner',
        group: 'image',
        hasMultiLinePrompts: true,
        // The deconstruction JSON is long; the Python captioner also enforces this floor.
        minNewTokens: 3072,
        defaults: {
            'config.process[0].caption.model_name_or_path': ['Qwen/Qwen3-VL-8B-Instruct', defaultNameOrPath],
            'config.process[0].caption.extensions': [extensionsImage, defaultExtensions],
            'config.process[0].caption.caption_prompt': [defaultIdeogramCaptionPrompt, undefined],
            'config.process[0].caption.max_res': [512, undefined],
            'config.process[0].caption.max_new_tokens': [4096, undefined],
            'config.process[0].caption.temperature': [ideogram4Preset.temperature, undefined],
            'config.process[0].caption.top_p': [ideogram4Preset.top_p, undefined],
            'config.process[0].caption.top_k': [ideogram4Preset.top_k, undefined],
            'config.process[0].caption.repetition_penalty': [ideogram4Preset.repetition_penalty, undefined],
        },
        name_or_path_options: [
            { value: 'Qwen/Qwen3-VL-2B-Instruct', label: 'Qwen/Qwen3-VL-2B-Instruct' },
            { value: 'Qwen/Qwen3-VL-4B-Instruct', label: 'Qwen/Qwen3-VL-4B-Instruct' },
            { value: 'Qwen/Qwen3-VL-8B-Instruct', label: 'Qwen/Qwen3-VL-8B-Instruct' },
            { value: 'Qwen/Qwen3-VL-30B-A3B-Instruct', label: 'Qwen/Qwen3-VL-30B-A3B-Instruct' },
        ],
        additionalSections: [
            'caption.caption_prompt',
            'caption.max_res',
            'caption.max_new_tokens',
            'caption.gen_params',
        ],
    },

].sort((a, b) => {
    // Sort by label, case-insensitive
    return a.label.localeCompare(b.label, undefined, { sensitivity: 'base' });
}) as any;

export const groupedCaptionerTypes: GroupedSelectOption[] = captionerTypes.reduce((acc, arch) => {
    const group = acc.find(g => g.label === arch.group);
    if (group) {
        group.options.push({ value: arch.name, label: arch.label });
    } else {
        acc.push({
            label: arch.group,
            options: [{ value: arch.name, label: arch.label }],
        });
    }
    return acc;
}, [] as GroupedSelectOption[]);

export const quantizationOptions: SelectOption[] = [
    { value: '', label: '- NONE -' },
    { value: 'float8', label: 'float8 (default)' },
    { value: 'uint7', label: '7 bit' },
    { value: 'uint6', label: '6 bit' },
    { value: 'uint5', label: '5 bit' },
    { value: 'uint4', label: '4 bit' },
    { value: 'uint3', label: '3 bit' },
    { value: 'uint2', label: '2 bit' },
];

export const maxResOptions: SelectOption[] = [
    { value: '256', label: '256' },
    { value: '512', label: '512 (default)' },
    { value: '768', label: '768' },
    { value: '1024', label: '1024' },
];
export const maxNewTokensOptions: SelectOption[] = [
    { value: '64', label: '64' },
    { value: '128', label: '128 (default)' },
    { value: '256', label: '256' },
    { value: '512', label: '512' },
    { value: '1024', label: '1024' },
    { value: '2048', label: '2048' },
    { value: '3072', label: '3072' },
    { value: '4096', label: '4096' },
    { value: '8192', label: '8192' },
];

export const defaultQtype = 'float8';