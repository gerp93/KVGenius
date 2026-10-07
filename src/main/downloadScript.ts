import * as path from 'path';
import { ManifestFeature } from '../shared/modelManifest';
import { DownloadItemInfo, DownloadPlanInfo } from '../shared/modelDownloads';
import { isExternalWebUrl } from '../shared/externalUrl';
import { fileState, ModelStatusReport } from '../shared/modelStatus';

/**
 * What to fetch: the files of these features that ComfyUI (or the models folder) does not have, and whose link is a
 * web address. A file that is in a subfolder is not fetched again - it is reported, and the user moves it.
 */
export function planDownloads(features: ManifestFeature[], report: ModelStatusReport, modelsDir: string): DownloadPlanInfo {
  const items: DownloadItemInfo[] = [];
  const inSubfolder: string[] = [];
  const seen = new Set<string>();
  for (const feature of features) {
    for (const file of feature.files) {
      const key = `${file.folder}/${file.file}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const state = fileState(file, report).state;
      if (state === 'in-subfolder') inSubfolder.push(file.file);
      if (state !== 'missing' || !file.url || !isExternalWebUrl(file.url)) continue;
      items.push({ label: file.role, url: file.url, folder: file.folder, fileName: file.file, destPath: path.join(modelsDir, file.folder, file.file) });
    }
  }
  return { items, inSubfolder, problem: null };
}

/** A string as one literal in a POSIX shell: nothing inside it is ever expanded or run. */
export function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

/** A string as one literal in PowerShell: single quotes expand nothing; only a quote itself needs doubling. */
export function powershellQuote(value: string): string {
  return `'${value.replace(/'/g, "''")}'`;
}

const HEADER = 'KVGenius model download';

/**
 * A script for macOS and Linux. Each file goes to `<name>.part` and is renamed when whole, so an interrupted run
 * is resumed by running the script again (`curl -C -` continues the partial file). Every path and link is a quoted
 * literal passed as an argument - none is ever spliced into a command.
 */
export function buildPosixScript(items: DownloadItemInfo[]): string {
  const calls = items.map((i) => `download ${shellQuote(i.url)} ${shellQuote(i.destPath)} ${shellQuote(i.label)}`).join('\n');
  return `#!/bin/sh
# ${HEADER}. Run it again to resume anything that was interrupted.
if ! command -v curl >/dev/null 2>&1; then
  echo "curl was not found. Install curl, or download the files by hand from the links below."
${items.map((i) => `  echo ${shellQuote(`  ${i.url}`)}`).join('\n')}
  printf 'Press Enter to close...'; read _unused || true
  exit 1
fi

failed=0
download() {
  url=$1; dest=$2; label=$3
  mkdir -p "$(dirname "$dest")"
  if [ -e "$dest" ]; then echo "Already there: $dest"; return 0; fi
  echo
  echo "Downloading $label"
  echo "  from $url"
  echo "    to $dest"
  if curl -L -C - --fail --retry 3 --progress-bar -o "$dest.part" "$url"; then
    mv "$dest.part" "$dest" && echo "  done"
  else
    echo "  FAILED - run this again to resume (a 404 means the link has moved)"
    failed=$((failed + 1))
  fi
}

echo ${shellQuote(`${HEADER}: ${items.length} file${items.length === 1 ? '' : 's'}`)}
${calls}

echo
if [ "$failed" -eq 0 ]; then echo "All done. Back in KVGenius, use Check Again on the Models page."; else echo "$failed file(s) did not finish. Run this again to resume."; fi
printf 'Press Enter to close...'; read _unused || true
# The exit status says whether every file finished.
[ "$failed" -eq 0 ]
`;
}

/**
 * The same for Windows, as a PowerShell script (curl.exe has shipped with Windows since 2018). It is started with
 * `-File`, never built into a command line, and every value is a single-quoted literal.
 */
export function buildPowershellScript(items: DownloadItemInfo[]): string {
  const calls = items
    .map((i) => `if (-not (Get-OneFile ${powershellQuote(i.url)} ${powershellQuote(i.destPath)} ${powershellQuote(i.label)})) { $failed++ }`)
    .join('\n');
  return `# ${HEADER}. Run it again to resume anything that was interrupted.
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
  Write-Host 'curl.exe was not found. Download the files by hand from the links below.'
${items.map((i) => `  Write-Host ${powershellQuote(`  ${i.url}`)}`).join('\n')}
  Read-Host 'Press Enter to close'
  exit 1
}

$failed = 0
function Get-OneFile([string]$Url, [string]$Dest, [string]$Label) {
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dest) | Out-Null
  if (Test-Path -LiteralPath $Dest) { Write-Host "Already there: $Dest"; return $true }
  Write-Host ''
  Write-Host "Downloading $Label"
  Write-Host "  from $Url"
  Write-Host "    to $Dest"
  & curl.exe -L -C - --fail --retry 3 -o "$Dest.part" $Url
  if ($LASTEXITCODE -eq 0) {
    Move-Item -LiteralPath "$Dest.part" -Destination $Dest
    Write-Host '  done'
    return $true
  }
  Write-Host '  FAILED - run this again to resume (a 404 means the link has moved)'
  return $false
}

Write-Host ${powershellQuote(`${HEADER}: ${items.length} file${items.length === 1 ? '' : 's'}`)}
${calls}

Write-Host ''
if ($failed -eq 0) { Write-Host 'All done. Back in KVGenius, use Check Again on the Models page.' } else { Write-Host "$failed file(s) did not finish. Run this again to resume." }
Read-Host 'Press Enter to close'
`;
}

export interface TerminalLaunch {
  command: string;
  args: string[];
}

/** Terminal programs tried on Linux, in order, and how each is told to run a command. */
const LINUX_TERMINALS: { command: string; args: (script: string) => string[] }[] = [
  { command: 'x-terminal-emulator', args: (s) => ['-e', 'sh', s] },
  { command: 'gnome-terminal', args: (s) => ['--', 'sh', s] },
  { command: 'konsole', args: (s) => ['-e', 'sh', s] },
  { command: 'xfce4-terminal', args: (s) => ['-x', 'sh', s] },
  { command: 'kitty', args: (s) => ['sh', s] },
  { command: 'alacritty', args: (s) => ['-e', 'sh', s] },
  { command: 'xterm', args: (s) => ['-e', 'sh', s] },
];

/** The file name the script is saved under on this platform. */
export function scriptFileName(platform: NodeJS.Platform): string {
  return platform === 'win32' ? 'download-models.ps1' : platform === 'darwin' ? 'download-models.command' : 'download-models.sh';
}

export function buildScript(platform: NodeJS.Platform, items: DownloadItemInfo[]): string {
  return platform === 'win32' ? buildPowershellScript(items) : buildPosixScript(items);
}

/**
 * The command that opens a terminal window running the script, or null when none can be found (the caller then
 * hands the script over to be run by hand). The script path is always its own argument, never part of a string.
 */
export function terminalLaunch(platform: NodeJS.Platform, scriptPath: string, isAvailable: (command: string) => boolean): TerminalLaunch | null {
  if (platform === 'win32') {
    return { command: 'cmd.exe', args: ['/c', 'start', '', 'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', scriptPath] };
  }
  if (platform === 'darwin') return { command: 'open', args: ['-a', 'Terminal', scriptPath] };
  const found = LINUX_TERMINALS.find((t) => isAvailable(t.command));
  return found ? { command: found.command, args: found.args(scriptPath) } : null;
}
