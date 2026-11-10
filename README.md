# tdmspy

A pure Python reader for National Instruments **TDMS** files, MIT licensed. Its only runtime dependency is numpy.

It is written from the format specification and tested against real files and
synthetic test cases, one for every special case of the format.

## Install

```sh
uv venv --allow-existing
uv sync --all-extras
```

## Usage

```python
import tdmspy

file = tdmspy.read("measurement.tdms")  # indexes the file; no channel data is read yet
for group in file.groups:  # in file order, not sorted
    for channel in group.channels:
        print(group.name, channel.name, channel.dtype, len(channel), channel.properties)

values = file["rawdata"]["Channel1"].read()  # numpy array, native byte order, NI scaling applied
raw = file["rawdata"]["Channel1"].read(scaled=False)
part = file["rawdata"]["Channel1"].read(start=1000, stop=2000)
```

The details (what is returned for each data type, what is refused, how truncated files are treated) are in
[`doc/design.md`](doc/design.md); every special case of the format the reader has to handle is listed with the files
that exercise it in [`doc/special-cases.md`](doc/special-cases.md).

## Tests

```sh
just test        # lint, then everything except the slow tests
just test-all    # including large files and long fuzz runs
just testcases      # rewrite the stored synthetic files in tests/data/synthetic
```

The tests are the specification of the reader:

- `tests/catalogue.py` lists every special case of the format. Add a case there first.
- `tests/builder.py` is a byte-level TDMS writer for the tests (interleaved, big-endian, several chunks, truncated,
  corrupt, every property type, ...). It does not use tdmspy.
- `tests/testcases.py` has one small file per special case, each with what it must read as (`Truth`). The files are
  stored in `tests/data/synthetic`; the long ones are built when a test needs them.
- `tests/data/real` are real files, with their identifying strings replaced by
  placeholders; `tests/test_real_files.py` checks them and `tests/test_privacy.py` keeps identifiers out of the test data.
- `tests/test_reader_*.py` are the tests of `tdmspy.read`: all test cases, the API, strict mode and scaling.
- `tests/test_fuzz.py` and `tests/test_robust.py` damage files and read large ones.

`just catalogue` regenerates `doc/special-cases.md`.

## Inspired by

[npTDMS](https://github.com/adamreeve/npTDMS), [pyTDMS](https://github.com/petebachant/pyTDMS) and
[ICOschema](https://github.com/ift-tuwien/ICOschema), which tdmspy was created to aid: ICOschema reads TDMS files
through it. No code of any of these projects is used here.

TDMS, LabVIEW and NI are trademarks of National Instruments. tdmspy is not affiliated with or endorsed by NI.

## Licence

MIT, see [LICENSE](LICENSE).
