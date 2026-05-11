import struct
def get_tags(filepath):
    with open(filepath, "rb") as fh:
        header = fh.read(8)
        endian = "<" if header[:2] == b"II" else ">"
        offset = struct.unpack(endian + "I", header[4:8])[0]
        fh.seek(offset)
        num_tags = struct.unpack(endian + "H", fh.read(2))[0]
        for _ in range(num_tags):
            tag, typ, count, value_offset = struct.unpack(endian + "HHII", fh.read(12))
            if tag == 270: # ImageDescription
                print("Found ImageDescription!")
                current = fh.tell()
                fh.seek(value_offset)
                desc = fh.read(count)
                print(desc[:500])
                fh.seek(current)

get_tags('../example-datasets/cif_example')