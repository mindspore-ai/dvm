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

    in_insn, pipe_cnt = False, 0
    with open(isa_file, 'r') as isa_file:
        for line in isa_file:
            if not in_insn:
                if line.startswith('enum vAccInsnID'):
                    print('extern const unsigned long int g_access_func_offset[] = {')
                    in_insn = True
                    pipe_cnt += 1
                elif line.startswith('enum vSimdInsnID'):
                    print('extern const unsigned long int g_simd_func_offset[] = {')
                    in_insn = True
                    pipe_cnt += 1
            else:
                stripped_line = line.strip()
                if not stripped_line.startswith("//"):
                    ins_name = extract_instruction_name(stripped_line)
                    if ins_name != None:
                        if ins_name.endswith("_NONE"):
                            in_insn = False
                            print('0\n};')
                            if pipe_cnt == 2:
                                break
                        else:
                            offset = function_address_map.get(ins_name, None)
                            if offset == None:
                                raise ValueError("Cannot find function for instruction: {}".format(ins_name))
                            print(function_address_map.get(ins_name, '0x0000') + ', // ' + ins_name)
    if pipe_cnt < 2:
        raise ValueError("Some pipe not found")
