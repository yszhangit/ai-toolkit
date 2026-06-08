/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { captionPromptsFolder } from '@/paths';

// Only plain-text prompt files are listed. Filename (sans extension) is the title.
const ALLOWED_EXT = new Set(['.txt', '.md']);

// Folder docs, not selectable prompts.
function isPromptFile(filename: string): boolean {
  if (filename.startsWith('.')) return false;
  if (!ALLOWED_EXT.has(path.extname(filename).toLowerCase())) return false;
  return titleFromFile(filename).toLowerCase() !== 'readme';
}

function ensureFolder(): string {
  if (!fs.existsSync(captionPromptsFolder)) {
    fs.mkdirSync(captionPromptsFolder, { recursive: true });
  }
  return captionPromptsFolder;
}

function titleFromFile(filename: string): string {
  return filename.replace(/\.(txt|md)$/i, '');
}

// GET /api/caption/prompts            -> [{ title, filename }]
// GET /api/caption/prompts?name=Foo   -> { title, content }
export async function GET(request: NextRequest) {
  try {
    const folder = ensureFolder();
    const name = request.nextUrl.searchParams.get('name');

    if (name) {
      // Read one prompt. Reduce to a basename so a malicious `name` can't escape
      // the folder via path separators or `..`.
      const safeBase = path.basename(name);
      const dirent = fs
        .readdirSync(folder, { withFileTypes: true })
        .find(d => d.isFile() && isPromptFile(d.name) && titleFromFile(d.name) === safeBase);
      if (!dirent) {
        return NextResponse.json({ error: 'Prompt not found' }, { status: 404 });
      }
      const content = fs.readFileSync(path.join(folder, dirent.name), 'utf-8');
      return NextResponse.json({ title: titleFromFile(dirent.name), content });
    }

    // List all prompts, sorted by title.
    const prompts = fs
      .readdirSync(folder, { withFileTypes: true })
      .filter(d => d.isFile() && isPromptFile(d.name))
      .map(d => ({ title: titleFromFile(d.name), filename: d.name }))
      .sort((a, b) => a.title.localeCompare(b.title, undefined, { sensitivity: 'base' }));

    return NextResponse.json(prompts);
  } catch (error) {
    console.error('Error listing caption prompts:', error);
    return NextResponse.json({ error: 'Failed to list caption prompts' }, { status: 500 });
  }
}
