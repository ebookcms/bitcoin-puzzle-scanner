# Bitcoin Puzzle Scanner

🚀 **GPU-accelerated Bitcoin private key scanner** using CUDA and CuPy to solve Bitcoin puzzles through optimized brute-force search.

[English](#english) | [Português](#português)

---

<a name="english"></a>
## 🇬🇧 English

### 📋 Features

- ✅ **Extreme Performance**: Uses NVIDIA GPU with CUDA for parallel processing
- ✅ **Professional CLI**: Complete command-line argument interface
- ✅ **Auto Puzzle Detection**: Automatically detects which puzzle you're searching
- ✅ **Automatic WIF Generation**: Generates Wallet Import Format for direct import
- ✅ **Smart FATBIN Selection**: Auto-detects GPU and loads optimized code
- ✅ **Flexible Ranges**: Accepts hex (with or without 0x) and decimal
- ✅ **Configurable Settings**: Presets and full customization
- ✅ **Real-time Statistics**: Progress, speed, ETA in single line
- ✅ **Robust Validation**: Checks all inputs before execution
- ✅ **Structured Output**: Results in TXT and JSON (append mode)
- ✅ **Verbose Mode**: Detailed stats at configurable intervals
- ✅ **Benchmark Mode**: Test GPU performance before search

### 🎯 What are Bitcoin Puzzles?

Bitcoin puzzles are challenges created by transferring Bitcoin to addresses with known public keys but unknown private keys within specific ranges (powers of 2). Solving them requires finding the private key through exhaustive search.

**Active Puzzles**: https://privatekeys.pw/puzzles/bitcoin-puzzle-tx

### 🛠️ Requirements

#### Hardware
- **GPU**: NVIDIA GPU with compute capability 7.0+ (Pascal or newer)
  - Recommended: RTX 2000, 3000, 4000 series
  - Minimum: GTX 2000 Ti or better

#### Software
- **OS**: Linux (Ubuntu 20.04+), Windows 10/11
- **Python**: 3.8 or higher
- **CUDA Toolkit**: 11.0 or higher
- **NVIDIA Driver**: Latest recommended

#### Python Libraries
```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

### 📦 Installation

#### 1. Clone Repository
```bash
git clone https://github.com/ebookcms/bitcoin-puzzle-scanner.git
cd bitcoin-puzzle-scanner
```

#### 2. Install Dependencies
```bash
[WINDOWS]
pip install cupy-cuda12x  # or cupy-cuda11x for CUDA 11

[LINUX]
python3 -m venv venv
source venv/bin/activate
pip install -U cupy-cuda12x
```

#### 3. CUDA Kernel already compilied
```bash
You are free to use the software without restrictions, provided you 
respect the original authorship. Authentication is required solely to 
prevent unauthorized redistribution or plagiarism of the code.
```

### 3.1 The kernel has been compiled from source and is 100% clean. 
```bash
There are no backdoors or hidden scripts.
```

## 3.2 You can use specific GPU
```bash
wrappers_sm_75.fatbin  # Turing (RTX 2000, GTX 1600)
wrappers_sm_80.fatbin  # Ampere (A100)
wrappers_sm_86.fatbin  # Ampere (RTX 3000)
wrappers_sm_89.fatbin  # Ada Lovelace (RTX 4000)
wrappers_sm_90.fatbin  # Hopper (H100)
```

#### 4. Verify Installation
```bash
python3 bitcoin_puzzle_scanner.py --version
python3 bitcoin_puzzle_scanner.py --benchmark-only -p 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798 -r 1:100
```

### 🚀 Quick Start

#### Basic Usage
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798 \
  -r 0x1:0xff
```

#### Search Puzzle #40 (Solved for demonstration)
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
  -r 0x8000000000:0xffffffffff
```

#### With Verbose Mode
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
  -r 0x8000000000:0xffffffffff \
  -v --batch-report 10
```

### 📖 Usage

```
usage: bitcoin_puzzle_scanner.py [-h] -p PUBLIC_KEY -r RANGE [options]

Bitcoin Puzzle Scanner - GPU-accelerated private key search

Required Arguments:
  -p PUBLIC_KEY         Compressed public key (66 hex chars, starting with 02/03)
  -r RANGE              Search range (hex or decimal)
                        Examples: 0x1:0xff, 1:255, 0x8000000000:0xffffffffff

GPU Configuration:
  --blocks N            Number of CUDA blocks (default: 1024)
  --threads N           Threads per block (default: 256)
  --iterations N        Iterations per thread (default: 100000)
  --preset NAME         Use preset configuration (fast/balanced/aggressive)
  --fatbin PATH         Path to CUDA fatbin file (auto-detected if not specified)
  --device N            CUDA device number (default: 0)

Behavior Options:
  --benchmark-only      Only run benchmark, do not search
  -v, --verbose         Show detailed stats at intervals
  --batch-report N      Show detailed stats every N batches (default: 10)
  --update-interval S   Progress update interval in seconds (default: 0.5)

File Options:
  --output-dir DIR      Directory for output files (default: current)

Info:
  --version             Show version
  -h, --help            Show this help message
```

### ⚙️ Configuration Presets

| Preset | Blocks | Threads | Iterations | Description |
|--------|--------|---------|------------|-------------|
| **fast** | 512 | 128 | 50,000 | Lower GPU usage, good for testing |
| **balanced** | 1,024 | 256 | 100,000 | Default, recommended for most GPUs |
| **aggressive** | 2,048 | 512 | 200,000 | Maximum GPU usage, best performance |

#### Using Presets
```bash
# Fast mode
python3 bitcoin_puzzle_scanner.py -p PUBKEY -r RANGE --preset fast

# Aggressive mode
python3 bitcoin_puzzle_scanner.py -p PUBKEY -r RANGE --preset aggressive
```

### 📊 Performance

#### Benchmark Results

| GPU | Speed (MKeys/s) | Puzzle #40 Time |
|-----|-----------------|-----------------|
| RTX 4090 | ~2,000 | ~4.5 min |
| RTX 3090 | ~1,400 | ~6.5 min |
| RTX 3080 | ~900 | ~10 min |
| RTX 3070 | ~600 | ~15 min |
| RTX 2080 Ti | ~500 | ~18 min |

*Benchmark: Puzzle #40 (549.76G keys)*

#### Run Benchmark
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
  -r 0x8000000000:0xffffffffff \
  --benchmark-only
```

### 📝 Examples

#### Example 1: Search Small Range (Testing)
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798 \
  -r 1:1000
```

#### Example 2: Search Puzzle #30
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 02f6a8eb18bd8d0667ecd36f1a4e8247e1a726e09a2e41a1d6e0e8f5a6bd1e6a7f \
  -r 0x20000000:0x3fffffff
```

#### Example 3: Custom GPU Configuration
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
  -r 0x8000000000:0xffffffffff \
  --blocks 2048 --threads 512 --iterations 200000
```

#### Example 4: Verbose Mode with Detailed Stats
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
  -r 0x8000000000:0xffffffffff \
  -v --batch-report 5
```

### 📤 Output

#### Console Output
```
================================================================================
BITCOIN PUZZLE SCANNER - GPU Accelerated
================================================================================

📍 Search Range:
   Start: 0x8000000000
   End:   0xffffffffff
   Total: 549.76G keys

🎯 Detected: Bitcoin Puzzle #40
   Coverage: 100.00% of puzzle range

✅ Auto-detected GPU fatbin: wrappers_sm_86.fatbin
🎮 GPU Device: 0 - NVIDIA GeForce RTX 3080 (CC 8.6)

🎯 Target Public Key: 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4

⚡ Running Benchmark...
   Speed: 889.32 MKeys/s
   Estimated time: 10.3m

================================================================================
🚀 STARTING SEARCH...
================================================================================

📊 Batch #10 | Progress: 42.91% | Speed: 887.62 MKeys/s | ETA: 5.9m

================================================================================
🎉🎉🎉 PRIVATE KEY FOUND! 🎉🎉🎉
================================================================================

🎯 Bitcoin Puzzle #40 SOLVED!

Private Key (Hex): 0xe9ae4933d6
Private Key (Dec): 1003651412950

🔑 WIF Generated:
   Compressed:   KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9aFJuCJDo5F6Jm7
   Uncompressed: 5HpHagT65TZzG1PH3CSu63k8DbpvD8s5ip4nEB6ikvy2duGEu2D

📊 Statistics:
   Batches: 18
   Keys scanned: 445.64G
   Time: 8.9m
   Avg speed: 837.94 MKeys/s

💾 Results saved:
   📄 puzzle40_key_found.txt
   📄 puzzle40_key_found.json
```

#### JSON Output
```json
{
  "timestamp": "2025-01-21T10:30:45.123456",
  "puzzle_number": 40,
  "private_key_hex": "0xe9ae4933d6",
  "private_key_dec": "1003651412950",
  "public_key": "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4",
  "wif_compressed": "KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9aFJuCJDo5F6Jm7",
  "wif_uncompressed": "5HpHagT65TZzG1PH3CSu63k8DbpvD8s5ip4nEB6ikvy2duGEu2D",
  "search_range_start": "0x8000000000",
  "search_range_end": "0xffffffffff",
  "keys_scanned": 445644800000,
  "time_elapsed_seconds": 534.5,
  "average_speed_mkeys": 837.94
}
```

### 🔧 Troubleshooting

#### CUDA_ERROR_NOT_FOUND
```
❌ ERRO ao carregar módulo CUDA: CUDA_ERROR_NOT_FOUND
```

**Solution**: Recompile CUDA kernel
```bash
bash compile.sh
```

#### No CUDA-capable device
```
CuPy error: No CUDA-capable device is detected
```

**Solution**: 
1. Verify NVIDIA driver: `nvidia-smi`
2. Verify CUDA: `nvcc --version`
3. Reinstall CuPy: `pip install --force-reinstall cupy-cuda12x`

#### Low Performance
```
Speed: 50 MKeys/s (expected ~800+)
```

**Solutions**:
1. Use aggressive preset: `--preset aggressive`
2. Close other GPU applications
3. Update NVIDIA drivers
4. Check GPU temperature/throttling

### 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 📄 License

You are free to use the software without restrictions, provided you respect the original authorship. 
Authentication is required solely to prevent unauthorized redistribution or plagiarism of the code.


### ⚠️ Disclaimer

This tool is for **educational and research purposes only**. 

- Only search for puzzles you have permission to solve
- Do not use for unauthorized access to wallets
- The authors are not responsible for misuse of this software

### 🙏 Acknowledgments

- Bitcoin Puzzle Transaction creator
- NVIDIA CUDA Team
- CuPy Development Team
- Bitcoin Community

### 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/bitcoin-puzzle-scanner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/bitcoin-puzzle-scanner/discussions)

---

<a name="português"></a>
## 🇵🇹 Português

### 📋 Características

- ✅ **Performance Extrema**: Utiliza GPU NVIDIA com CUDA para processamento paralelo
- ✅ **Interface CLI Profissional**: Argumentos de linha de comando completos
- ✅ **Auto-detecção de Puzzles**: Detecta automaticamente qual puzzle você está procurando
- ✅ **Geração Automática de WIF**: Gera Wallet Import Format para importação direta
- ✅ **Seleção Inteligente de FATBIN**: Auto-detecta GPU e carrega código otimizado
- ✅ **Ranges Flexíveis**: Aceita hex (com ou sem 0x) e decimal
- ✅ **Configurações Personalizáveis**: Presets e customização total
- ✅ **Estatísticas em Tempo Real**: Progresso, velocidade, ETA em linha única
- ✅ **Validação Robusta**: Verifica todos os inputs antes de executar
- ✅ **Output Estruturado**: Resultados em TXT e JSON (modo append)
- ✅ **Modo Verbose**: Estatísticas detalhadas em intervalos configuráveis
- ✅ **Modo Benchmark**: Testa performance da GPU antes de buscar

### 🎯 O que são Bitcoin Puzzles?

Bitcoin puzzles são desafios criados transferindo Bitcoin para endereços com chaves públicas conhecidas mas chaves privadas desconhecidas dentro de ranges específicos (potências de 2). Resolvê-los requer encontrar a chave privada através de busca exaustiva.

**Puzzles Ativos**: https://privatekeys.pw/puzzles/bitcoin-puzzle-tx

### 🛠️ Requisitos

#### Hardware
- **GPU**: NVIDIA GPU com compute capability 7.0+
  - Recomendado: RTX 2000, 3000, 4000 series
  - Mínimo: GTX 1050 Ti ou melhor

#### Software
- **SO**: Linux (Ubuntu 20.04+), Windows 10/11
- **Python**: 3.8 ou superior
- **CUDA Toolkit**: 11.0 ou superior
- **Driver NVIDIA**: Última versão recomendada

#### Bibliotecas Python
```bash
# Para CUDA 12.x
pip install cupy-cuda12x

# Para CUDA 11.x
pip install cupy-cuda11x
```

### 📦 Instalação

#### 1. Clonar Repositório
```bash
git clone https://github.com/ebookcms/bitcoin-puzzle-scanner.git
cd bitcoin-puzzle-scanner
```

#### 2. Instalar Dependências
```bash
pip install cupy-cuda12x  # ou cupy-cuda11x para CUDA 11
```

#### 3. CUDA Kernel já compilado
```bash
Você é livre para usar o software sem restrições, desde que respeite a 
autoria original. A autenticação é exigida exclusivamente para evitar a 
redistribuição não autorizada ou o plágio do código.
```

### 3.1 O kernel foi compilado a partir do código-fonte e está 100% limpo. 
```bash
Não existem backdoors ou scripts ocultos.
```

## 3.2 You can use specific GPU
```bash
wrappers_sm_75.fatbin  # Turing (RTX 2000, GTX 1600)
wrappers_sm_80.fatbin  # Ampere (A100)
wrappers_sm_86.fatbin  # Ampere (RTX 3000)
wrappers_sm_89.fatbin  # Ada Lovelace (RTX 4000)
wrappers_sm_90.fatbin  # Hopper (H100)
```

#### 4. Verificar Instalação
```bash
python3 bitcoin_puzzle_scanner.py --version
python3 bitcoin_puzzle_scanner.py --benchmark-only -p 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798 -r 1:100
```

### 🚀 Início Rápido

#### Uso Básico
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798 \
  -r 0x1:0xff
```

#### Buscar Puzzle #40 (Resolvido para demonstração)
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
  -r 0x8000000000:0xffffffffff
```

#### Com Modo Verbose
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
  -r 0x8000000000:0xffffffffff \
  -v --batch-report 10
```

### 📖 Uso

```
uso: bitcoin_puzzle_scanner.py [-h] -p PUBLIC_KEY -r RANGE [opções]

Bitcoin Puzzle Scanner - Busca de chave privada acelerada por GPU

Argumentos Obrigatórios:
  -p PUBLIC_KEY         Chave pública comprimida (66 chars hex, começa com 02/03)
  -r RANGE              Range de busca (hex ou decimal)
                        Exemplos: 0x1:0xff, 1:255, 0x8000000000:0xffffffffff

Configuração GPU:
  --blocks N            Número de blocos CUDA (padrão: 1024)
  --threads N           Threads por bloco (padrão: 256)
  --iterations N        Iterações por thread (padrão: 100000)
  --preset NAME         Usar configuração preset (fast/balanced/aggressive)
  --fatbin PATH         Caminho para arquivo CUDA fatbin (auto-detectado)
  --device N            Número do dispositivo CUDA (padrão: 0)

Opções de Comportamento:
  --benchmark-only      Apenas executar benchmark, não buscar
  -v, --verbose         Mostrar estatísticas detalhadas em intervalos
  --batch-report N      Mostrar stats detalhadas a cada N batches (padrão: 10)
  --update-interval S   Intervalo de atualização em segundos (padrão: 0.5)

Opções de Arquivo:
  --output-dir DIR      Diretório para arquivos de saída (padrão: atual)

Info:
  --version             Mostrar versão
  -h, --help            Mostrar esta mensagem de ajuda
```

### ⚙️ Presets de Configuração

| Preset | Blocos | Threads | Iterações | Descrição |
|--------|--------|---------|-----------|-----------|
| **fast** | 512 | 128 | 50.000 | Menor uso de GPU, bom para testes |
| **balanced** | 1.024 | 256 | 100.000 | Padrão, recomendado para maioria das GPUs |
| **aggressive** | 2.048 | 512 | 200.000 | Uso máximo de GPU, melhor performance |

#### Usando Presets
```bash
# Modo rápido
python3 bitcoin_puzzle_scanner.py -p PUBKEY -r RANGE --preset fast

# Modo agressivo
python3 bitcoin_puzzle_scanner.py -p PUBKEY -r RANGE --preset aggressive
```

### 📊 Performance

#### Resultados de Benchmark

| GPU | Velocidade (MKeys/s) | Tempo Puzzle #40 |
|-----|----------------------|------------------|
| RTX 4090 | ~2.000 | ~4,5 min |
| RTX 3090 | ~1.400 | ~6,5 min |
| RTX 3080 | ~900 | ~10 min |
| RTX 3070 | ~600 | ~15 min |
| RTX 2080 Ti | ~500 | ~18 min |

*Benchmark: Puzzle #40 (549,76G chaves)*

#### Executar Benchmark
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
  -r 0x8000000000:0xffffffffff \
  --benchmark-only
```

### 📝 Exemplos

#### Exemplo 1: Buscar Range Pequeno (Teste)
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798 \
  -r 1:1000
```

#### Exemplo 2: Buscar Puzzle #30
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 02f6a8eb18bd8d0667ecd36f1a4e8247e1a726e09a2e41a1d6e0e8f5a6bd1e6a7f \
  -r 0x20000000:0x3fffffff
```

#### Exemplo 3: Configuração GPU Customizada
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
  -r 0x8000000000:0xffffffffff \
  --blocks 2048 --threads 512 --iterations 200000
```

#### Exemplo 4: Modo Verbose com Stats Detalhadas
```bash
python3 bitcoin_puzzle_scanner.py \
  -p 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 \
  -r 0x8000000000:0xffffffffff \
  -v --batch-report 5
```

### 🔧 Solução de Problemas

#### CUDA_ERROR_NOT_FOUND
```
❌ ERRO ao carregar módulo CUDA: CUDA_ERROR_NOT_FOUND
```

**Solução**: Recompilar kernel CUDA
```bash
bash compile.sh
```

#### Nenhum dispositivo CUDA detectado
```
Erro CuPy: No CUDA-capable device is detected
```

**Solução**: 
1. Verificar driver NVIDIA: `nvidia-smi`
2. Verificar CUDA: `nvcc --version`
3. Reinstalar CuPy: `pip install --force-reinstall cupy-cuda12x`

#### Performance Baixa
```
Velocidade: 50 MKeys/s (esperado ~800+)
```

**Soluções**:
1. Usar preset agressivo: `--preset aggressive`
2. Fechar outras aplicações que usam GPU
3. Atualizar drivers NVIDIA
4. Verificar temperatura/throttling da GPU

### 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se livre para submeter um Pull Request.

1. Fork o repositório
2. Crie sua feature branch (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

### 📄 Licença

Você é livre para usar o software sem restrições, desde que respeite a autoria original. 
A autenticação é exigida exclusivamente para evitar a redistribuição não autorizada ou o plágio do código..

### ⚠️ Aviso Legal

Esta ferramenta é **apenas para fins educacionais e de pesquisa**.

- Apenas busque puzzles que você tem permissão para resolver
- Não use para acesso não autorizado a carteiras
- Os autores não são responsáveis pelo uso indevido deste software

### 🙏 Agradecimentos

- Criador da Bitcoin Puzzle Transaction
- Equipe NVIDIA CUDA
- Equipe de Desenvolvimento CuPy
- Comunidade Bitcoin

### 📞 Suporte

- **Issues**: [GitHub Issues](https://github.com/seuusuario/bitcoin-puzzle-scanner/issues)
- **Discussões**: [GitHub Discussions](https://github.com/seuusuario/bitcoin-puzzle-scanner/discussions)

---

**Made with ❤️ for the Bitcoin community**
