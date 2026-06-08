import React, { useEffect, useState } from 'react';
import {
  Checkbox,
  CreatableSelectInput,
  FormGroup,
  NumberInput,
  SelectInput,
  TextAreaInput,
  TextInput,
} from '@/components/formInputs';
import { apiClient } from '@/utils/api';
import { CaptionJobConfig } from '@/types';
import { handleCaptionerTypeChange } from '@/helpers/captionJobConfig';
import {
  captionerTypes,
  defaultQtype,
  groupedCaptionerTypes,
  maxNewTokensOptions,
  maxResOptions,
  quantizationOptions,
} from '@/helpers/captionOptions';

type Props = {
  jobConfig: CaptionJobConfig;
  setJobConfig: (value: any, key?: string) => void;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  showGPUSelect: boolean;
};

const CaptionSimpleJob: React.FC<Props> = ({ jobConfig, setJobConfig, gpuIDs, setGpuIDs, gpuList, showGPUSelect }) => {
  const selectedCaptionOption = captionerTypes.find(option => option.name === jobConfig.config.process[0].type);
  const additionalSections = selectedCaptionOption?.additionalSections || [];
  const minNewTokens = selectedCaptionOption?.minNewTokens ?? 0;
  const newTokensOptions = maxNewTokensOptions.filter(option => parseInt(option.value) >= minNewTokens);
  const caption = jobConfig.config.process[0].caption;

  // User-authored system prompts from the caption_prompts/ folder.
  const [savedPrompts, setSavedPrompts] = useState<{ title: string; filename: string }[]>([]);
  useEffect(() => {
    if (!additionalSections.includes('caption.caption_prompt')) return;
    apiClient
      .get('/api/caption/prompts')
      .then(res => setSavedPrompts(Array.isArray(res.data) ? res.data : []))
      .catch(() => setSavedPrompts([]));
  }, [additionalSections.includes('caption.caption_prompt')]);

  const loadSavedPrompt = (title: string) => {
    if (!title) return;
    apiClient
      .get(`/api/caption/prompts?name=${encodeURIComponent(title)}`)
      .then(res => {
        if (typeof res.data?.content === 'string') {
          setJobConfig(res.data.content.trim(), 'config.process[0].caption.caption_prompt');
        }
      })
      .catch(error => console.error('Error loading saved prompt:', error));
  };

  return (
    <div className="text-sm text-gray-400">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <div>
          <SelectInput
            label="Captioner Type"
            value={jobConfig.config.process[0].type}
            onChange={value => {
              handleCaptionerTypeChange(jobConfig.config.process[0].type, value, jobConfig, setJobConfig);
            }}
            options={groupedCaptionerTypes}
          />
        </div>
        {showGPUSelect && (
          <div>
            <SelectInput
              label="GPU ID"
              value={`${gpuIDs}`}
              onChange={value => setGpuIDs(value)}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
            />
          </div>
        )}
      </div>
      <div className="mt-4">
        <CreatableSelectInput
          label="Name or Path"
          value={jobConfig.config.process[0].caption.model_name_or_path}
          docKey="config.process[0].caption.model_name_or_path"
          onChange={(value: string | null) => {
            if (value?.trim() === '') {
              value = null;
            }
            setJobConfig(value, 'config.process[0].caption.model_name_or_path');
          }}
          placeholder=""
          options={selectedCaptionOption?.name_or_path_options || []}
          required
        />
      </div>
      {additionalSections.includes('caption.model_name_or_path2') && (
        <div className="mt-4">
          <CreatableSelectInput
            label="Name or Path 2"
            value={jobConfig.config.process[0].caption.model_name_or_path2 || ''}
            onChange={(value: string | null) => {
              if (value?.trim() === '') {
                value = null;
              }
              setJobConfig(value, 'config.process[0].caption.model_name_or_path2');
            }}
            placeholder=""
            options={selectedCaptionOption?.name_or_path2_options || []}
          />
        </div>
      )}
      {additionalSections.includes('caption.fixed_caption') && (
        <div className="mt-4">
          <TextInput
            label="Fixed Caption"
            value={jobConfig.config.process[0].caption.fixed_caption || ''}
            onChange={value => {
              if (value?.trim() === '') {
                //@ts-ignore
                value = undefined;
              }
              setJobConfig(value, 'config.process[0].caption.fixed_caption');
            }}
            placeholder="Enter fixed caption (if you want the same caption for all audio files)"
          />
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <div>
          <SelectInput
            label="Quantize"
            value={jobConfig.config.process[0].caption.quantize ? jobConfig.config.process[0].caption.qtype : ''}
            onChange={value => {
              if (value === '') {
                setJobConfig(false, 'config.process[0].caption.quantize');
                value = defaultQtype;
              } else {
                setJobConfig(true, 'config.process[0].caption.quantize');
              }
              setJobConfig(value, 'config.process[0].caption.qtype');
            }}
            options={quantizationOptions}
          />
          <div className="mt-4">
            <CreatableSelectInput
              label="Caption Extension"
              value={jobConfig.config.process[0].caption.caption_extension || 'txt'}
              onChange={value => {
                setJobConfig(value, 'config.process[0].caption.caption_extension');
              }}
              options={[
                { value: 'txt', label: 'txt' },
                { value: 'json', label: 'json' },
                { value: 'caption', label: 'caption' },
              ]}
            />
          </div>
          {additionalSections.includes('caption.max_res') && (
            <div className="mt-4">
              <SelectInput
                label="Max Resolution"
                value={`${jobConfig.config.process[0].caption.max_res || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.max_res');
                  }
                }}
                options={maxResOptions}
              />
            </div>
          )}
          {additionalSections.includes('caption.max_new_tokens') && (
            <div className="mt-4">
              <SelectInput
                label="Max New Tokens"
                value={`${jobConfig.config.process[0].caption.max_new_tokens || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.max_new_tokens');
                  }
                }}
                options={newTokensOptions}
              />
            </div>
          )}
        </div>
        <div>
          <FormGroup label="Options">
            <Checkbox
              label="Low VRAM"
              checked={jobConfig.config.process[0].caption.low_vram}
              onChange={value => setJobConfig(value, 'config.process[0].caption.low_vram')}
            />
            <Checkbox
              label="Recaption"
              checked={jobConfig.config.process[0].caption.recaption}
              onChange={value => setJobConfig(value, 'config.process[0].caption.recaption')}
            />
          </FormGroup>
        </div>
      </div>
      {additionalSections.includes('caption.gen_params') && (
        <div className="mt-4">
          <FormGroup label="Sampling Parameters">
            <p className="text-xs text-gray-500 mb-2">
              Leave a field blank to use the model&apos;s built-in default.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <NumberInput
                label="Temperature"
                value={caption.temperature ?? null}
                min={0}
                onChange={value => setJobConfig(value, 'config.process[0].caption.temperature')}
                placeholder="default"
              />
              <NumberInput
                label="Top P"
                value={caption.top_p ?? null}
                min={0}
                max={1}
                onChange={value => setJobConfig(value, 'config.process[0].caption.top_p')}
                placeholder="default"
              />
              <NumberInput
                label="Top K"
                value={caption.top_k ?? null}
                min={0}
                onChange={value => setJobConfig(value, 'config.process[0].caption.top_k')}
                placeholder="default"
              />
              <NumberInput
                label="Repetition Penalty"
                value={caption.repetition_penalty ?? null}
                min={0}
                onChange={value => setJobConfig(value, 'config.process[0].caption.repetition_penalty')}
                placeholder="default"
              />
            </div>
          </FormGroup>
        </div>
      )}
      {additionalSections.includes('caption.caption_prompt') && (
        <div className="mt-4">
          {savedPrompts.length > 0 && (
            <SelectInput
              label="Load System Prompt"
              value=""
              onChange={value => loadSavedPrompt(value)}
              options={[
                { value: '', label: '- Select a saved prompt -' },
                ...savedPrompts.map(p => ({ value: p.title, label: p.title })),
              ]}
            />
          )}
          <div className="mt-2">
            <TextAreaInput
              label="Caption Prompt"
              value={caption.caption_prompt || ''}
              onChange={value => {
                setJobConfig(value, 'config.process[0].caption.caption_prompt');
              }}
              placeholder="Enter caption prompt"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default CaptionSimpleJob;
