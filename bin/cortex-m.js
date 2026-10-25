#!/usr/bin/env node

/**
 * CORTEX-M Universal CLI Wrapper (Node.js / npm / npx runner)
 * 
 * Allows Cortex-M and Cortex Studio to be installed and run globally:
 *   npx cortex-m
 *   npm install -g . && cortex-m studio
 *   cortex-m chat
 *   cortex-m code
 */

const { spawn, spawnSync, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT_DIR = path.resolve(__dirname, '..');

// Helper to test if a command exists and runs
function testCommand(cmd, args = ['--version']) {
  try {
    const res = spawnSync(cmd, args, { stdio: 'ignore', windowsHide: true });
    return res.status === 0;
  } catch (err) {
    return false;
  }
}

// Locate appropriate Python interpreter
function findPython() {
  // 1. Check local virtual environments first
  const venvCandidates = [
    path.join(ROOT_DIR, '.venv', 'Scripts', 'python.exe'),
    path.join(ROOT_DIR, '.venv', 'bin', 'python'),
    path.join(ROOT_DIR, '..', '.venv-mimo', 'Scripts', 'python.exe'),
    path.join(ROOT_DIR, '..', '.venv-mimo', 'bin', 'python'),
    path.join(ROOT_DIR, '.venv-mimo', 'Scripts', 'python.exe'),
    path.join(ROOT_DIR, '.venv-mimo', 'bin', 'python')
  ];

  for (const cand of venvCandidates) {
    if (fs.existsSync(cand)) {
      return cand;
    }
  }

  // 2. Check process.env.PYTHON
  if (process.env.PYTHON && testCommand(process.env.PYTHON)) {
    return process.env.PYTHON;
  }

  // 3. Check system python3, python, or py
  const systemCandidates = ['python3', 'python', 'py'];
  for (const cand of systemCandidates) {
    if (testCommand(cand)) {
      return cand;
    }
  }

  return null;
}

function verifyAndSetupDependencies(pythonBin) {
  // Check if cortex_m and torch are installed
  const checkCmd = spawnSync(pythonBin, ['-c', 'import torch, cortex_m; print("OK")'], {
    cwd: ROOT_DIR,
    encoding: 'utf-8',
    windowsHide: true
  });

  if (checkCmd.status !== 0 || !checkCmd.stdout || !checkCmd.stdout.includes('OK')) {
    console.log('\n======================================================');
    console.log(' [Cortex-M] First-time setup: installing dependencies...');
    console.log('======================================================');
    const installRes = spawnSync(pythonBin, ['-m', 'pip', 'install', '-e', '.'], {
      cwd: ROOT_DIR,
      stdio: 'inherit'
    });
    if (installRes.status !== 0) {
      console.warn('[Cortex-M] Note: pip install -e . returned non-zero. Attempting requirements.txt...');
      spawnSync(pythonBin, ['-m', 'pip', 'install', '-r', 'requirements.txt'], {
        cwd: ROOT_DIR,
        stdio: 'inherit'
      });
    }
  }
}

function main() {
  const args = process.argv.slice(2);
  const pythonBin = findPython();

  if (!pythonBin) {
    console.error('\n[Error] Python 3.9+ was not found on your system.');
    console.error('Please install Python from https://python.org or your package manager and try again.\n');
    process.exit(1);
  }

  const subCommand = args[0] ? args[0].toLowerCase() : 'chat';

  if (subCommand === '--help' || subCommand === '-h' || subCommand === 'help') {
    console.log(`
========================================================================
 CORTEX-M: Headless Neural Memory Runtime & Grounded Cognition (CLI)
========================================================================

Usage:
  npx cortex-m chat           Start interactive neural conversational terminal (default)
  npx cortex-m server         Launch headless JSON REST API server
  npx cortex-m substrate      Biological synaptic memory CLI (learn, recall)
  npx cortex-m bench          Run complete scientific comparative benchmark suite
  npx cortex-m test           Run complete test suite (129 tests)
  npx cortex-m setup          Install / verify Python backend dependencies

Global Install:
  npm install -g .            Install 'cortex-m' CLI command globally on system
  cortex-m chat               Start conversational interface anywhere
`);
    process.exit(0);
  }

  if (subCommand === 'setup') {
    console.log(`[Cortex-M] Setting up dependencies with ${pythonBin}...`);
    verifyAndSetupDependencies(pythonBin);
    console.log('\n[Cortex-M] Setup complete! Run `npx cortex-m chat` to start.\n');
    process.exit(0);
  }

  // Quick verify
  verifyAndSetupDependencies(pythonBin);

  let pyArgs = [];
  const remainingArgs = args.slice(1);

  if (subCommand === 'server' || subCommand === 'api') {
    pyArgs = ['-m', 'apps.api.server', ...remainingArgs];
  } else if (subCommand === 'chat' || subCommand === 'lm') {
    const chatScript = path.join(ROOT_DIR, 'scripts', 'cortex_lm_chat.py');
    pyArgs = [chatScript, ...remainingArgs];
  } else if (subCommand === 'code' || subCommand === 'swarm') {
    const codeScript = path.join(ROOT_DIR, 'scripts', 'cortex_code.py');
    pyArgs = [codeScript, ...remainingArgs];
  } else if (subCommand === 'substrate' || subCommand === 'memory') {
    const cliScript = path.join(ROOT_DIR, 'scripts', 'cortex_cli.py');
    pyArgs = [cliScript, ...remainingArgs];
  } else if (subCommand === 'test') {
    pyArgs = ['-m', 'unittest', 'discover', '-s', 'tests', ...remainingArgs];
  } else if (subCommand === 'bench' || subCommand === 'benchmark') {
    const benchScript = path.join(ROOT_DIR, 'benchmarks', 'run_comparison_benchmark.py');
    pyArgs = [benchScript, ...remainingArgs];
  } else {
    // Pass everything through to cortex unified CLI
    pyArgs = ['-m', 'cortex_m.cli', ...args];
  }

  const child = spawn(pythonBin, pyArgs, {
    cwd: ROOT_DIR,
    stdio: 'inherit',
    env: {
      ...process.env,
      PYTHONPATH: path.join(ROOT_DIR, 'src') + (process.env.PYTHONPATH ? path.delimiter + process.env.PYTHONPATH : '')
    }
  });

  child.on('exit', (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
    } else {
      process.exit(code || 0);
    }
  });
}

main();
