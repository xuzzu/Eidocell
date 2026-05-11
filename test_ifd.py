import struct
def find_first_image_chunk(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(8)
        endian = '<' if header[:2] == b'II' else '>'
        offset = struct.unpack(endian + 'I', header[4:8])[0]
        while offset != 0:
            f.seek(offset)
            num_tags = struct.unpack(endian + 'H', f.read(2))[0]
            tags = {}
            for i in range(num_tags):
                tag_data = f.read(12)
                tag, typ, count, value_offset = struct.unpack(endian + 'HHII', tag_data)
                tags[tag] = value_offset
            offset = struct.unpack(endian + 'I', f.read(4))[0]
            comp = tags.get(259)
            if comp in [30817, 30818]:
                width = tags.get(256)
                height = tags.get(257)
                return width, height
    return None, None

print("CIF:", find_first_image_chunk('/Users/xuzzu/Documents/DataSpell Projects/eidocell-v2/example-datasets/cif_example'))
print("RIF:", find_first_image_chunk('/Users/xuzzu/Documents/DataSpell Projects/eidocell-v2/example-datasets/rif_example'))
