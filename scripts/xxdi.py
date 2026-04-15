#!/usr/bin/env python3
"""
Minimal xxd -i replacement for DVM build.
Converts binary file to C array format.

Usage: python3 scripts/xxd_i.py <input_file>
Output is written to stdout.
"""

import sys
import os


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_file>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]

    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' not found", file=sys.stderr)
        sys.exit(1)

    # Read binary file
    with open(input_file, 'rb') as f:
        data = f.read()

    # Generate variable name (same as xxd -i)
    var_name = os.path.basename(input_file).replace('.', '_').replace('-', '_')

    # Output C array
    print(f"unsigned char {var_name}[] = {{")

    for i in range(0, len(data), 12):
        chunk = data[i:i + 12]
        hex_vals = ', '.join(f'0x{b:02x}' for b in chunk)
        if i + 12 < len(data):
            print(f"  {hex_vals},")
        else:
            print(f"  {hex_vals}")

    print("};")
    print(f"unsigned int {var_name}_len = {len(data)};")


if __name__ == '__main__':
    main()