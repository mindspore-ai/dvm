# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import sys
import re

def process_address(address):
    """
    Processes a hexadecimal address, ensuring it's within a valid range.
    
    Parameters:
        address (str): The address to process.
        
    Returns:
        str: Processed address in hexadecimal format.
        
    Raises:
        ValueError: If the address exceeds the valid range.
    """
    head = int(address[:-4], 16)
    if head != 0:
        raise ValueError("Address exceeds 0xFFFF.")
    return '0x' + address[-4:]

def extract_instruction_name(line):
    """
    Extracts the instruction name from a line of code.
    
    Parameters:
        line (str): A line of C++ code.
        
    Returns:
        str or None: The extracted instruction name, or None if not found.
    """
    pattern = r'\b(V_[\w]+)\b'
    match = re.search(pattern, line)
    if match:
        return match.group(1)
    else:
        return None

if __name__ == '__main__':
    # Get arguments
    isa_file = sys.argv[1]

    load_pipe, store_pipe, simd_pipe = 0, 1, 2
    pipe_names = ("load", "store", "simd")
    insn_names = ([], [], [])

    # Extract all SIMD instructions from the isa file
    with open(isa_file, 'r') as isa_file:
        cur_pipe = -1
        for line in isa_file:
            if cur_pipe == -1:
                if line.startswith('enum vSimdInsnID'):
                    cur_pipe = simd_pipe
                elif line.startswith('enum vLoadInsnID'):
                    cur_pipe = load_pipe
                elif line.startswith('enum vStoreInsnID'):
                    cur_pipe = store_pipe
            else:
                stripped_line = line.strip()
                if not stripped_line.startswith("//"):
                    ins_name = extract_instruction_name(stripped_line)
                    if ins_name != None:
                        if ins_name.endswith("_NONE"):
                            cur_pipe = -1
                        else:
                            insn_names[cur_pipe].append(ins_name)

    # Process symbol table to find function addresses
    try:
        function_address_map = {}
        pattern = r'^([0-9a-f]+) .* D_(V_[\w]+)$'
        vmain_pattern = r"^0000000000000000 .* vmain_mix_aiv$"
        while True:
            line = input()
            match = re.search(pattern, line)
            if match:
                address, ins_name = match.groups()
                function_address_map[ins_name] = process_address(address)
    except EOFError:
        pass

    # Ensure all instructions have been mapped
    total_insn_num = len(insn_names[0]) + len(insn_names[1]) + len(insn_names[2])
    if len(function_address_map) != total_insn_num:
        raise ValueError("Mismatch between function addresses and instructions: {} : {}".format(
            len(function_address_map), total_insn_num))

    for i in range(3):
        # Output function addresses
        print('extern const unsigned long int g_{}_func_offset[] = {{'.format(pipe_names[i]))
        for ins_name in insn_names[i]:
            print(function_address_map.get(ins_name, '0x0000') + ', // ' + ins_name)
        print('0\n};')
