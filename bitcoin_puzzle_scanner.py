#!/usr/bin/env python3
"""
Bitcoin Puzzle Scanner - Professional GPU-based private key search tool
Searches for Bitcoin private keys by matching public key X coordinates

example puzzle#30:
> python bitcoin_puzzle_scanner.py -p 030d282cf2ff536d2c42f105d0b8588821a915dc3f9a05bd98bb23af67a2e92a5b -r 20000000:3fffffff

example puzzle#40:
python bitcoin_puzzle_scanner.py -p 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 -r 0x8000000000:0xffffffffff

example Puzlle#125
full range: 10000000000000000000000000000000:1fffffffffffffffffffffffffffffff -> for test I will start -> 1c533b6bb7f0804e0996020000000000
python bitcoin_puzzle_scanner.py -p 0233709eb11e0d4439a729f21c2c443dedb727528229713f0065721ba8fa46f00e -r 1c533b6bb7f0804e0996020000000000:1c533b6bb7f0804e0996ffffffffffff  -v --batch-report 10
"""

import cupy as cp
import argparse
import sys
import time
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_FATBIN = "wrappers.fatbin"
SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

# Bitcoin puzzle ranges (2^(n-1) to 2^n - 1)
PUZZLE_RANGES = {
    1:   (0x1, 0x1),
    2:   (0x2, 0x3),
    3:   (0x4, 0x7),
    4:   (0x8, 0xf),
    5:   (0x10, 0x1f),
    10:  (0x200, 0x3ff),
    15:  (0x4000, 0x7fff),
    20:  (0x80000, 0xfffff),
    25:  (0x1000000, 0x1ffffff),
    30:  (0x20000000, 0x3fffffff),
    35:  (0x400000000, 0x7ffffffff),
    40:  (0x8000000000, 0xffffffffff),
    45:  (0x100000000000, 0x1fffffffffff),
    50:  (0x2000000000000, 0x3ffffffffffff),
    55:  (0x40000000000000, 0x7fffffffffffff),
    60:  (0x800000000000000, 0xfffffffffffffff),
    65:  (0x10000000000000000, 0x1ffffffffffffffff),
    70:  (0x200000000000000000, 0x3fffffffffffffffff),
    75:  (0x4000000000000000000, 0x7ffffffffffffffffff),
    80:  (0x80000000000000000000, 0xffffffffffffffffffff),
    85:  (0x1000000000000000000000, 0x1fffffffffffffffffffff),
    90:  (0x20000000000000000000000, 0x3ffffffffffffffffffffff),
    95:  (0x400000000000000000000000, 0x7fffffffffffffffffffffff),
    100: (0x8000000000000000000000000, 0xfffffffffffffffffffffffff),
    105: (0x100000000000000000000000000, 0x1ffffffffffffffffffffffffff),
    110: (0x2000000000000000000000000000, 0x3fffffffffffffffffffffffffff),
    115: (0x40000000000000000000000000000, 0x7ffffffffffffffffffffffffffff),
    120: (0x800000000000000000000000000000, 0xffffffffffffffffffffffffffffff),
    125: (0x10000000000000000000000000000000, 0x1fffffffffffffffffffffffffffffff),
    130: (0x200000000000000000000000000000000, 0x3ffffffffffffffffffffffffffffffff),
    135: (0x4000000000000000000000000000000000, 0x7fffffffffffffffffffffffffffffffff),
    140: (0x80000000000000000000000000000000000, 0xfffffffffffffffffffffffffffffffffff),
    145: (0x1000000000000000000000000000000000000, 0x1ffffffffffffffffffffffffffffffffffff),
    150: (0x10000000000000000000000000000000000000, 0x1fffffffffffffffffffffffffffffffffffff),
    155: (0x400000000000000000000000000000000000000, 0x7ffffffffffffffffffffffffffffffffffffff),
    160: (0x8000000000000000000000000000000000000000, 0xffffffffffffffffffffffffffffffffffffffff)
}

# Configuration presets
PRESETS = {
    "fast": {
        "blocks": 512,
        "threads": 128,
        "iterations": 50000,
        "description": "Fast - Lower GPU usage, good for testing"
    },
    "balanced": {
        "blocks": 1024,
        "threads": 256,
        "iterations": 100000,
        "description": "Balanced - Recommended default configuration"
    },
    "aggressive": {
        "blocks": 2048,
        "threads": 512,
        "iterations": 200000,
        "description": "Aggressive - Maximum GPU usage"
    }
}

# ============================================================================
# AUTHORIZATION
# ============================================================================

def initialize_and_auth(module):
    """
    Initializes and authorizes the CUDA kernel.
    This function MUST be called before using the scanner.
    When it is called, the kernel is unlocked and displays its logo.
    
    Args:
        module: Loaded CuPy RawModule
    
    Returns:
        True if authorization is successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("🔓 STARTING KERNEL AUTHORIZATION...")
    print("=" * 80)
    
    try:
        # Get signature function
        sig_kernel = module.get_function("get_kernel_signature")
        
        # Buffer to receive ASCII signature (2048 bytes)
        d_buffer = cp.zeros(2048, dtype=cp.int8)
        
        # Execute kernel - activate is_authorized = 1
        sig_kernel((1,), (1,), (d_buffer,))
        cp.cuda.Stream.null.synchronize()
        
        # Get data from GPU
        raw_data = d_buffer.get()
        
        # Convert to string
        # int8 can have negative values, so we use % 256
        byte_data = bytes([x % 256 for x in raw_data if x != 0])
        signature = byte_data.decode('utf-8', errors='ignore')
        
        # Display logo/signature
        print()
        print(signature)
        print()
        print("=" * 80)
        print("✅ KERNEL AUTHORIZED AND ACTIVATED SUCCESSFULLY!")
        print("=" * 80)
        print()
        
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ ERROR INITIALIZING AUTHORIZATION: {e}")
        print("=" * 80)
        print()
        print("⚠️  The kernel has NOT been activated!")
        print("    All operations will return 0xDEADBEEF")
        print()
        import traceback
        traceback.print_exc()
        return False


def check_authorization_result(result):
    """
    Check if the kernel result indicates a lack of authorization.
    
    Args:
        result: Value returned by kernel
    
    Returns:
        True if authorized (valid result)
        False if unauthorized (0xDEADBEEF)
    """
    if result == 0xDEADBEEF:
        print()
        print("=" * 80)
        print("⚠️  KERNEL NOT AUTHORIZED!")
        print("=" * 80)
        print()
        print("The kernel returned 0xDEADBEEF indicating a lack of authorization.")
        print()
        print("To activate the kernel, call:")
        print("  initialize_and_auth(module)")
        print()
        print("Before executing any search operation.")
        print("=" * 80)
        print()
        return False
    
    return True



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def base58_encode(data: bytes) -> str:
    """Encode bytes to Base58"""
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    
    # Convert bytes to integer
    num = int.from_bytes(data, 'big')
    
    # Convert to base58
    encoded = ''
    while num > 0:
        num, remainder = divmod(num, 58)
        encoded = alphabet[remainder] + encoded
    
    # Add '1' for each leading zero byte
    for byte in data:
        if byte == 0:
            encoded = '1' + encoded
        else:
            break
    
    return encoded


def private_key_to_wif(private_key: int, compressed: bool = True) -> str:
    """
    Convert private key to WIF (Wallet Import Format)
    
    Args:
        private_key: Private key as integer
        compressed: True for compressed WIF, False for uncompressed
    
    Returns:
        WIF string
    """
    # Convert private key to 32 bytes
    private_key_bytes = private_key.to_bytes(32, byteorder='big')
    
    # Add prefix (0x80 for mainnet)
    extended_key = b'\x80' + private_key_bytes
    
    # Add compression flag if compressed
    if compressed:
        extended_key += b'\x01'
    
    # Double SHA256 hash for checksum
    hash1 = hashlib.sha256(extended_key).digest()
    hash2 = hashlib.sha256(hash1).digest()
    checksum = hash2[:4]
    
    # Append checksum
    final_key = extended_key + checksum
    
    # Encode in Base58
    wif = base58_encode(final_key)
    
    return wif


def detect_puzzle_number(start: int, end: int) -> Optional[int]:
    """
    Detect which Bitcoin puzzle this range corresponds to
    
    Args:
        start: Start of range
        end: End of range
    
    Returns:
        Puzzle number or None if not a known puzzle
    """
    for puzzle_num, (puzzle_start, puzzle_end) in PUZZLE_RANGES.items():
        # Check if search range overlaps with or is within puzzle range
        if start >= puzzle_start and end <= puzzle_end:
            return puzzle_num
        # Also detect if search range contains the puzzle range
        if start <= puzzle_start and end >= puzzle_end:
            return puzzle_num
    
    return None


def detect_gpu_architecture() -> str:
    """
    Detect GPU compute capability and return appropriate fatbin name
    
    Returns:
        Fatbin filename for the detected GPU
    """
    try:
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        major = props['major']
        minor = props['minor']
        
        # Map compute capability to architecture
        compute_cap = f"{major}{minor}"
        
        arch_map = {
            '75': 'sm_75',  # Turing (RTX 2000, GTX 1600)
            '80': 'sm_80',  # Ampere (A100)
            '86': 'sm_86',  # Ampere (RTX 3000)
            '89': 'sm_89',  # Ada Lovelace (RTX 4000)
            '90': 'sm_90',  # Hopper (H100)
        }
        
        arch = arch_map.get(compute_cap, 'sm_86')  # Default to sm_86
        return f"wrappers_{arch}.fatbin"
        
    except Exception:
        # If detection fails, return default
        return "wrappers.fatbin"


def validate_hex(value: str, expected_length: Optional[int] = None) -> str:
    """Validates hexadecimal string"""
    value = value.strip()
    if value.startswith("0x") or value.startswith("0X"):
        value = value[2:]
    
    if not all(c in "0123456789abcdefABCDEF" for c in value):
        raise argparse.ArgumentTypeError(f"Valor hexadecimal inválido: {value}")
    
    if expected_length and len(value) != expected_length:
        raise argparse.ArgumentTypeError(
            f"Comprimento esperado: {expected_length}, recebido: {len(value)}"
        )
    
    return value.lower()


def parse_public_key(pubkey_str: str) -> int:
    """Parse and validate compressed public key"""
    pubkey_str = validate_hex(pubkey_str, 66)
    
    prefix = pubkey_str[:2]
    if prefix not in ("02", "03"):
        raise ValueError(f"Prefixo de chave pública inválido: {prefix} (esperado: 02 ou 03)")
    
    x_coord = pubkey_str[2:]
    return int(x_coord, 16)


def parse_range(range_str: str) -> Tuple[int, int]:
    """Parse string de range no formato START:END (accepts hex with or without 0x)"""
    try:
        parts = range_str.split(":")
        if len(parts) != 2:
            raise ValueError("Format must be START:END")
        
        start_str, end_str = parts
        
        # Removes spaces
        start_str = start_str.strip()
        end_str = end_str.strip()
        
        # Helper function to convert string to int
        def str_to_int(s: str) -> int:
            s = s.strip()
            # If it starts with 0x or 0X, it is hexadecimal.
            if s.lower().startswith('0x'):
                return int(s, 16)
            # If it contains only hex digits (without 0x), try hex first.
            elif all(c in '0123456789abcdefABCDEF' for c in s):
                # If it has more than 10 digits, it is probably hex.
                if len(s) > 10:
                    return int(s, 16)
                # If it contains letters a-f, it is definitely hex
                elif any(c in 'abcdefABCDEF' for c in s):
                    return int(s, 16)
                # Otherwise, try decimal first, then hex if it fails
                else:
                    try:
                        return int(s, 10)
                    except ValueError:
                        return int(s, 16)
            # If it doesn't look like hex, try decimal
            else:
                return int(s, 10)
        
        start = str_to_int(start_str)
        end = str_to_int(end_str)
        
        if start >= end:
            raise ValueError("START must be less than END")
        
        if start < 1 or end > SECP256K1_ORDER:
            raise ValueError(f"Range must be between 1 and {hex(SECP256K1_ORDER)}")
        
        return start, end
        
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid range '{range_str}': {e}")


def to_u64_array(val_int: int) -> cp.ndarray:
    """Converts integer to 4x uint64 array (Little Endian)"""
    return cp.array([
        (val_int >> 0)   & 0xFFFFFFFFFFFFFFFF,
        (val_int >> 64)  & 0xFFFFFFFFFFFFFFFF,
        (val_int >> 128) & 0xFFFFFFFFFFFFFFFF,
        (val_int >> 192) & 0xFFFFFFFFFFFFFFFF
    ], dtype=cp.uint64)


def format_number(n: float) -> str:
    """Format numbers in a readable way"""
    if n >= 1e15:
        return f"{n/1e15:.2f}P"
    elif n >= 1e12:
        return f"{n/1e12:.2f}T"
    elif n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return f"{n:,.0f}"


def format_hash_rate(rate: float) -> str:
    """Formats hash rate"""
    if rate >= 1e9:
        return f"{rate/1e9:.2f} GKeys/s"
    elif rate >= 1e6:
        return f"{rate/1e6:.2f} MKeys/s"
    elif rate >= 1e3:
        return f"{rate/1e3:.2f} KKeys/s"
    else:
        return f"{rate:.2f} Keys/s"


def format_duration(seconds: float) -> str:
    """Format time duration"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        days = int(seconds / 86400)
        return f"{days}d {(seconds % 86400)/3600:.1f}h"


# ============================================================================
# SCANNER CLASS
# ============================================================================

class BitcoinPuzzleScanner:
    """Main scanner class"""
    
    def __init__(self, args):
        self.args = args
        self.module = None
        self.kernel = None
        self.target_x = None
        self.range_start = None
        self.range_end = None
        self.total_keys = 0
        self.keys_per_batch = 0
        self.puzzle_number = None
        self.compressed = True  # Assume compressed by default (puzzles are compressed)
        
        # Statistics
        self.total_scanned = 0
        self.start_time = None
        self.batch_count = 0
        
    def initialize(self):
        """Initializes GPU and loads kernel"""
        print("=" * 80)
        print("BITCOIN PUZZLE SCANNER - GPU Accelerated")
        print("=" * 80)
        
        # Parse range first to detect puzzle
        try:
            self.range_start, self.range_end = parse_range(self.args.range)
            self.total_keys = self.range_end - self.range_start
            
            # Detect puzzle
            self.puzzle_number = detect_puzzle_number(self.range_start, self.range_end)
            
            print(f"\n📍 Search Range:")
            print(f"   Start: {hex(self.range_start)}")
            print(f"   End:   {hex(self.range_end)}")
            print(f"   Total: {format_number(self.total_keys)} keys")
            
            if self.puzzle_number:
                print(f"\n🎯 Detected: Bitcoin Puzzle #{self.puzzle_number}")
                puzzle_start, puzzle_end = PUZZLE_RANGES[self.puzzle_number]
                coverage = ((min(self.range_end, puzzle_end) - max(self.range_start, puzzle_start)) / 
                           (puzzle_end - puzzle_start + 1) * 100)
                print(f"   Coverage: {coverage:.2f}% of puzzle range")
            
        except Exception as e:
            print(f"\n❌ ERROR: Invalid range: {e}")
            return False
        
        # Detects GPU and selects appropriate fatbin
        if self.args.fatbin == DEFAULT_FATBIN:
            # Auto-detect fatbin based on GPU
            detected_fatbin = detect_gpu_architecture()
            if Path(detected_fatbin).exists():
                self.args.fatbin = detected_fatbin
                print(f"\n✅ Auto-detected GPU fatbin: {detected_fatbin}")
            elif Path(DEFAULT_FATBIN).exists():
                print(f"\n⚠️  Specific fatbin not found, using default: {DEFAULT_FATBIN}")
            else:
                # List available fatbins
                fatbins = list(Path('.').glob('wrappers*.fatbin'))
                if fatbins:
                    print(f"\n⚠️  Default fatbin not found. Available fatbins:")
                    for fb in fatbins:
                        print(f"   - {fb.name}")
                    print(f"\nUsing: {fatbins[0].name}")
                    self.args.fatbin = str(fatbins[0])
        
        # Load CUDA module
        fatbin_path = Path(self.args.fatbin)
        if not fatbin_path.exists():
            print(f"\n❌ ERROR: {self.args.fatbin} not found!")
            print(f"\n❌ Your device is not supported")
            return False
        
        try:
            self.module = cp.RawModule(path=str(fatbin_path))
            initialize_and_auth(self.module)
            self.kernel = self.module.get_function('range_scan_kernel')
            print(f"✅ CUDA module loaded: {fatbin_path}")
        except Exception as e:
            print(f"\n❌ ERROR loading CUDA module: {e}")
            return False
        
        # Prepare GPU
        device = cp.cuda.Device(self.args.device)
        device.use()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        gpu_name = props['name'].decode()
        compute_cap = f"{props['major']}.{props['minor']}"
        print(f"🎮 GPU Device: {device.id} - {gpu_name} (CC {compute_cap})")
        
        # Parse public key
        try:
            self.target_x = parse_public_key(self.args.public_key)
            print(f"\n🎯 Target Public Key: {self.args.public_key}")
            print(f"   X-coordinate: {hex(self.target_x)}")
            
            # Detects if it is compressed
            prefix = self.args.public_key[:2]
            self.compressed = prefix in ('02', '03')
            print(f"   Type: {'Compressed' if self.compressed else 'Uncompressed'}")
            
        except Exception as e:
            print(f"\n❌ ERROR: Invalid public key: {e}")
            return False
        
        # Configuration
        print(f"\n⚙️  Configuration:")
        print(f"   Blocks: {self.args.blocks:,}")
        print(f"   Threads/Block: {self.args.threads:,}")
        print(f"   Iterations/Thread: {self.args.iterations:,}")
        
        self.keys_per_batch = self.args.blocks * self.args.threads * self.args.iterations
        print(f"   Keys/Batch: {format_number(self.keys_per_batch)}")
        
        if self.args.preset:
            print(f"   Preset: {self.args.preset}")
        
        return True
    
    def scan_range_gpu(self, start_k: int) -> Optional[int]:
        """Scans a range using GPU"""
        start_k_gpu = to_u64_array(start_k)
        target_x_gpu = to_u64_array(self.target_x)
        found_idx = cp.zeros(1, dtype=cp.uint64)
        
        self.kernel(
            (self.args.blocks,), 
            (self.args.threads,),
            (start_k_gpu, target_x_gpu, cp.uint64(self.args.iterations), found_idx)
        )
        cp.cuda.Stream.null.synchronize()
        
        result = int(found_idx[0])
        
        # Checks if kernel is unauthorized
        if result == 0xDEADBEEF:
            check_authorization_result(result)
            sys.exit(1)
        
        if result > 0:
            return start_k + result - 1
        return None
    
    def benchmark(self):
        """Runs performance benchmark"""
        print("\n⚡ Running Benchmark...")
        
        test_iterations = 10000
        test_start = self.range_start
        
        # Warmup
        self.scan_range_gpu(test_start)
        
        # Teste real
        t0 = time.perf_counter()
        self.scan_range_gpu(test_start)
        t1 = time.perf_counter()
        
        elapsed = t1 - t0
        keys_tested = self.args.blocks * self.args.threads * test_iterations
        keys_per_sec = keys_tested / elapsed if elapsed > 0 else 0
        
        print(f"   Keys tested: {format_number(keys_tested)} in {elapsed:.3f}s")
        print(f"   Speed: {format_hash_rate(keys_per_sec)}")
        
        if keys_per_sec > 0 and self.total_keys > 0:
            total_seconds = self.total_keys / keys_per_sec
            if total_seconds < 365.25 * 24 * 3600:
                print(f"   Estimated time: {format_duration(total_seconds)}")
            else:
                years = total_seconds / (365.25 * 24 * 3600)
                if years > 1e9:
                    print(f"   Estimated time: {years:.2e} years")
                else:
                    print(f"   Estimated time: {years:,.0f} years")
        
        return keys_per_sec
    
    def save_result(self, private_key: int, stats: dict):
        """Saves result to file (append mode)"""
        
        try:
            # Generate WIF
            wif_compressed = private_key_to_wif(private_key, compressed=True)
            wif_uncompressed = private_key_to_wif(private_key, compressed=False)
            
            print(f"\n🔑 WIF Generated:")
            print(f"   Compressed:   {wif_compressed}")
            print(f"   Uncompressed: {wif_uncompressed}")
            
        except Exception as e:
            print(f"\n⚠️  Warning: Could not generate WIF: {e}")
            wif_compressed = "ERROR"
            wif_uncompressed = "ERROR"
        
        # Determines filename based on the puzzle
        if self.puzzle_number:
            base_name = f"puzzle{self.puzzle_number}_key_found"
        else:
            base_name = "key_found"
        
        txt_file = f"{base_name}.txt"
        json_file = f"{base_name}.json"
        
        # Result data
        result_data = {
            "found": True,
            "puzzle_number": self.puzzle_number,
            "private_key_hex": hex(private_key),
            "private_key_dec": str(private_key),
            "wif_compressed": wif_compressed,
            "wif_uncompressed": wif_uncompressed,
            "public_key": self.args.public_key,
            "public_key_type": "compressed" if self.compressed else "uncompressed",
            "target_x": hex(self.target_x),
            "timestamp": datetime.now().isoformat(),
            "search_range": {
                "start": hex(self.range_start),
                "end": hex(self.range_end)
            },
            "statistics": stats
        }
        
        try:
            # Savr/Append JSON
            existing_results = []
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            # Tries to load as a list
                            try:
                                existing_results = json.loads(content)
                                if not isinstance(existing_results, list):
                                    existing_results = [existing_results]
                            except json.JSONDecodeError:
                                # If it is not valid JSON, starts a new list
                                existing_results = []
                except Exception as e:
                    print(f"   ⚠️  Could not read existing JSON: {e}")
                    existing_results = []
            
            existing_results.append(result_data)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 JSON saved: {json_file}")
            
        except Exception as e:
            print(f"\n❌ Error saving JSON: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Append texto legível
            with open(txt_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"BITCOIN PUZZLE SCANNER - PRIVATE KEY FOUND\n")
                f.write("=" * 80 + "\n\n")
                
                if self.puzzle_number:
                    f.write(f"Bitcoin Puzzle #{self.puzzle_number}\n\n")
                
                f.write(f"Private Key (Hex): {hex(private_key)}\n")
                f.write(f"Private Key (Dec): {private_key}\n\n")
                
                f.write(f"WIF (Compressed):   {wif_compressed}\n")
                f.write(f"WIF (Uncompressed): {wif_uncompressed}\n\n")
                
                f.write(f"Public Key: {self.args.public_key}\n")
                f.write(f"Public Key Type: {'Compressed' if self.compressed else 'Uncompressed'}\n")
                f.write(f"Target X: {hex(self.target_x)}\n\n")
                
                f.write(f"Search Range:\n")
                f.write(f"  Start: {hex(self.range_start)}\n")
                f.write(f"  End:   {hex(self.range_end)}\n\n")
                
                f.write(f"Timestamp: {datetime.now()}\n\n")
                
                f.write(f"Statistics:\n")
                f.write(f"  Batches: {stats['batches']:,}\n")
                f.write(f"  Keys scanned: {stats['total_scanned']:,}\n")
                f.write(f"  Time elapsed: {format_duration(stats['elapsed'])}\n")
                f.write(f"  Average speed: {format_hash_rate(stats['avg_speed'])}\n")
                f.write("=" * 80 + "\n\n")
            
            print(f"💾 TXT saved: {txt_file}")
            
        except Exception as e:
            print(f"\n❌ Error saving TXT: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n✅ Results saved successfully!")
        print(f"   📄 {txt_file}")
        print(f"   📄 {json_file}")
        
        # Informações de importação
        if wif_compressed != "ERROR":
            print(f"\n🔑 For wallet import:")
            if self.compressed:
                print(f"   Use WIF (Compressed): {wif_compressed}")
            else:
                print(f"   Use WIF (Uncompressed): {wif_uncompressed}")
            print(f"\n   💡 Both WIF formats saved to files")
    
    def run(self):
        """Executa busca principal"""
        if not self.initialize():
            return 1
        
        # Benchmark
        speed = self.benchmark()
        
        if self.args.benchmark_only:
            print("\n✅ Benchmark completed")
            return 0
        
        # Search
        print("\n" + "=" * 80)
        print("🚀 STARTING SEARCH...")
        print("=" * 80)
        print()  # Blank line for progress
        
        current_k = self.range_start
        self.start_time = time.time()
        last_update_time = time.time()
        
        try:
            while current_k < self.range_end:
                self.batch_count += 1
                
                current_time = time.time()
                elapsed = current_time - self.start_time
                
                # Updates display only if the minimum interval has passed
                should_update = (current_time - last_update_time) >= self.args.update_interval
                
                if should_update:
                    progress = (self.total_scanned / self.total_keys * 100) if self.total_keys > 0 else 0
                    avg_speed = self.total_scanned / elapsed if elapsed > 0 else 0
                    
                    # Calculates ETA
                    eta_str = ""
                    if avg_speed > 0:
                        remaining = (self.total_keys - self.total_scanned) / avg_speed
                        eta_str = f"ETA: {format_duration(remaining)}"
                    
                    # ========================================================================
                    # DIFFERENTIATES REPORT BATCHES FROM NORMAL ONES
                    # ========================================================================
                    
                    # If verbose AND report batch: shows complete detailed stats
                    if self.args.verbose and self.batch_count % self.args.batch_report == 0:
                        print()  # New line (clears previous \r)
                        print(f"\n{'─'*80}")
                        print(f"📊 Detailed Stats - Batch #{self.batch_count:,}")
                        print(f"   Progress: {progress:.6f}%")
                        print(f"   Current Key: {hex(current_k)}")
                        print(f"   Keys Scanned: {self.total_scanned:,} / {self.total_keys:,}")
                        print(f"   Time Elapsed: {format_duration(elapsed)}")
                        print(f"   Average Speed: {format_hash_rate(avg_speed)}")
                        if avg_speed > 0:
                            print(f"   ETA: {format_duration(remaining)}")
                        print(f"{'─'*80}\n")
                    
                    else:
                        # Batches normais: linha única de progresso
                        progress_line = (
                            f"\r📊 Batch #{self.batch_count:,} | "
                            f"Progress: {progress:.4f}% | "
                            f"Current: {hex(current_k)} | "
                            f"Scanned: {format_number(self.total_scanned)}/{format_number(self.total_keys)} | "
                            f"Time: {format_duration(elapsed)} | "
                            f"Speed: {format_hash_rate(avg_speed)}"
                        )
                        
                        if eta_str:
                            progress_line += f" | {eta_str}"
                        
                        # Fills with spaces to clear the previous line
                        terminal_width = 120
                        progress_line = progress_line.ljust(terminal_width)
                        
                        # Prints without line break
                        print(progress_line, end='', flush=True)
                    
                    last_update_time = current_time

                
                # Scan
                result = self.scan_range_gpu(current_k)
                
                if result:
                    # New line before showing result (since we are using \r)
                    print()
                    
                    elapsed_final = time.time() - self.start_time
                    
                    # Generate WIF
                    wif_compressed = private_key_to_wif(result, compressed=True)
                    wif_uncompressed = private_key_to_wif(result, compressed=False)
                    
                    print("\n" + "=" * 80)
                    print("🎉🎉🎉 PRIVATE KEY FOUND! 🎉🎉🎉")
                    print("=" * 80)
                    
                    if self.puzzle_number:
                        print(f"\n🎯 Bitcoin Puzzle #{self.puzzle_number} SOLVED!")
                    
                    print(f"\nPrivate Key (Hex): {hex(result)}")
                    print(f"Private Key (Dec): {result}")
                    
                    print(f"\n🔑 WIF for Wallet Import:")
                    print(f"   Compressed:   {wif_compressed}")
                    print(f"   Uncompressed: {wif_uncompressed}")
                    
                    if self.compressed:
                        print(f"\n   💡 This is a compressed address - use: {wif_compressed}")
                    else:
                        print(f"\n   💡 This is an uncompressed address - use: {wif_uncompressed}")
                    
                    stats = {
                        "batches": self.batch_count,
                        "total_scanned": self.total_scanned,
                        "elapsed": elapsed_final,
                        "avg_speed": self.total_scanned / elapsed_final if elapsed_final > 0 else 0
                    }
                    
                    print(f"\n📊 Statistics:")
                    print(f"   Batches: {stats['batches']:,}")
                    print(f"   Keys scanned: {format_number(stats['total_scanned'])}")
                    print(f"   Time: {format_duration(stats['elapsed'])}")
                    print(f"   Avg speed: {format_hash_rate(stats['avg_speed'])}")
                    
                    self.save_result(result, stats)
                    print("=" * 80)
                    return 0
                
                current_k += self.keys_per_batch
                self.total_scanned += self.keys_per_batch
        
        except KeyboardInterrupt:
            print()  # New line before showing message
            print(f"\n\n⚠️  Interrupted by user!")
            elapsed = time.time() - self.start_time
            print(f"   Progress: {(self.total_scanned/self.total_keys)*100:.4f}%")
            print(f"   Last position: {hex(current_k)}")
            print(f"   Elapsed: {format_duration(elapsed)}")
            return 130
        
        except Exception as e:
            print()  # New line before showing error
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        print()  # New line before final message
        print(f"\n" + "=" * 80)
        print(f"❌ Search complete. Private key not found in range.")
        print("=" * 80)
        return 1


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_parser():
    """Creates argument parser"""
    parser = argparse.ArgumentParser(
        description="Bitcoin Puzzle Scanner - GPU-accelerated private key search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search (hex with 0x)
  %(prog)s -p 0233709eb11e0d4439a729f21c2c443dedb727528229713f0065721ba8fa46f00e \\
           -r 0x8000000000:0xffffffffff

  # Hex without 0x also works
  %(prog)s -p 02Public_Key_Here -r 8000000000:ffffffffff

  # Decimal range
  %(prog)s -p 02Public_Key_Here -r 1000000:2000000

  # Puzzle #40 (auto-detected)
  %(prog)s -p 02PublicKey -r 0x8000000000:0xffffffffff --preset balanced

  # With custom GPU settings
  %(prog)s -p 02Public_Key_Here -r 1000000:2000000 -b 2048 -t 512 -i 200000

  # Using preset configuration
  %(prog)s -p 02PublicKey -r START:END --preset aggressive

  # Benchmark only
  %(prog)s -p 02PublicKey -r START:END --benchmark-only

  # Verbose mode
  %(prog)s -p 02PublicKey -r START:END -v

Notes:
  - Range accepts hex (with or without 0x) and decimal formats
  - Puzzle number is auto-detected from range
  - Output files are named: puzzleN_key_found.txt/json
  - WIF (Wallet Import Format) is generated for direct import
  - Fatbin is auto-selected based on GPU (or specify with --fatbin)

Presets available:
  fast       - Low GPU usage, good for testing
  balanced   - Recommended default configuration
  aggressive - Maximum GPU utilization

For more information: https://github.com/ebookcms
        """
    )

    # Mandatory arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '-p', '--public-key', '--pub',
        required=True,
        metavar='PUBKEY',
        help='Compressed public key (66 hex chars, starts with 02 or 03)'
    )
    required.add_argument(
        '-r', '--range',
        required=True,
        metavar='START:END',
        help='Search range (format: START:END, hex or decimal)'
    )
    
    # Configuration GPU
    gpu_group = parser.add_argument_group('GPU configuration')
    gpu_group.add_argument(
        '-d', '--device',
        type=int,
        default=0,
        metavar='N',
        help='GPU device ID (default: 0)'
    )
    gpu_group.add_argument(
        '-b', '--blocks',
        type=int,
        default=1024,
        metavar='N',
        help='Number of GPU blocks (default: 1024)'
    )
    gpu_group.add_argument(
        '-t', '--threads',
        type=int,
        default=256,
        metavar='N',
        help='Threads per block (default: 256)'
    )
    gpu_group.add_argument(
        '-i', '--iterations',
        type=int,
        default=100000,
        metavar='N',
        help='Iterations per thread (default: 100000)'
    )
    gpu_group.add_argument(
        '--preset',
        choices=['fast', 'balanced', 'aggressive'],
        help='Use preset configuration (overrides -b, -t, -i)'
    )
    
    # Arquivos
    files_group = parser.add_argument_group('file options')
    files_group.add_argument(
        '--fatbin',
        default=DEFAULT_FATBIN,
        metavar='FILE',
        help=f'Path to CUDA fatbin file (default: auto-detect or {DEFAULT_FATBIN})'
    )
    
    # Comportamento
    behavior_group = parser.add_argument_group('behavior options')
    behavior_group.add_argument(
        '--benchmark-only',
        action='store_true',
        help='Only run benchmark, do not search'
    )
    behavior_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output (show every batch)'
    )
    behavior_group.add_argument(
        '--batch-report',
        type=int,
        default=10,
        metavar='N',
        help='Report detailed progress every N batches in verbose mode (default: 10)'
    )
    behavior_group.add_argument(
        '--update-interval',
        type=float,
        default=0.5,
        metavar='SEC',
        help='Update interval in seconds for live progress (default: 0.5)'
    )
    
    # Info
    parser.add_argument(
        '--version',
        action='version',
        version='Bitcoin Puzzle Scanner v1.2.1'
    )
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Applies preset if specified
    if args.preset:
        preset = PRESETS[args.preset]
        args.blocks = preset["blocks"]
        args.threads = preset["threads"]
        args.iterations = preset["iterations"]
    
    # Validate Configuration
    if args.blocks < 1 or args.threads < 1 or args.iterations < 1:
        print("❌ ERROR: blocks, threads, and iterations must be > 0")
        return 1
    
    if args.batch_report < 1:
        print("❌ ERROR: batch-report must be > 0")
        return 1
    
    if args.update_interval <= 0:
        print("❌ ERROR: update-interval must be > 0")
        return 1
    
    # Executa scanner
    scanner = BitcoinPuzzleScanner(args)
    return scanner.run()


if __name__ == "__main__":
    sys.exit(main())
