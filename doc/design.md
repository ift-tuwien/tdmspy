# Design and contract of tdmspy

This is what `tdmspy.read` has to do. The tests (`tests/`) check every statement here; where this file and a test
disagree, one of them is a bug. The list of special cases of the format is in [special-cases.md](special-cases.md).

## Goals and non-goals

- Read TDMS files (version 4712 and 4713) of any size, including those written by LabVIEW and other
  libraries; correctly in every layout the format allows that real files use.
- Pure Python and numpy, MIT licensed: no native code, no dependency beyond numpy.
- Fast enough to open a file of a million segments and to read a 1 GB channel without copying it more than once.
- Honest: what is not supported is an error that says so, never silently wrong data.
- Non-goals: writing TDMS files, DAQmx raw data, `.tdms_index` files, LabVIEW `.tdx` / TDM, pandas or HDF5 export.

## The API

```python
import tdmspy

file = tdmspy.read(path, strict=False)  # path: str or os.PathLike
```

`read` reads the structure of the file (every segment's lead-in and metadata) and keeps no channel data in memory.
The file is opened again for every `Channel.read`, so a `TdmsFile` holds no open file handle and can be used from
several threads.

| Object | Members |
|---|---|
| `TdmsFile` | `path`, `properties` (dict), `groups` (list, file order), `truncated` (bool); `file["name"]` returns the group (`KeyError` if missing), `"name" in file`, iterating gives the groups |
| `Group` | `name`, `properties`, `channels` (list, file order); `group["name"]`, `"name" in group` |
| `Channel` | `name`, `properties`, `dtype` (a native numpy dtype, `None` for a channel without a data type), `len(channel)`, `read(start=0, stop=None, *, scaled=True)` |
| `Timestamp` | `seconds` since 1904-01-01 UTC, `fraction` in units of 2**-64 s; `to_datetime64(unit)`, `from_datetime64` |

Order: groups and channels are in the order of their **first appearance** in the file (a channel's path appearing
counts for its group as well), never sorted. Properties are in the order they were first written; a later segment
that rewrites a property changes its value, not its position.

Names are exactly as stored. `''` in a path is one apostrophe; `/` inside quotes is part of the name.

### Property values

Python values: `int` for all integer types, `float` for single, double and the with-unit variants (a single
precision value is converted exactly), `bool`, `str`, `complex`, and `Timestamp` for timestamps (both parts kept
exactly). The width of a property is not kept.

### Channel data

`read` returns a numpy array in **native byte order**, whatever the file has, and a copy the caller may change.

| TDMS type | numpy |
|---|---|
| int8 ... int64, uint8 ... uint64 | the same |
| single, double | float32, float64 (NaN, infinities and -0.0 as stored) |
| boolean | bool |
| string | object array of `str` |
| timestamp | `datetime64[us]`, UTC, truncated to the microsecond (never rounded up) |
| complex single, double | complex64, complex128 |
| none (the channel never has a data type) | `dtype` is `None`, length 0, `read()` returns an empty array |
| extended float, anything else | `UnsupportedFeatureError` when the file is opened |

`start` and `stop` count values, like a numpy slice with non-negative bounds (negative bounds are a `ValueError`;
`stop` beyond the end is cut). A slice reads only the bytes it needs.

### Scaling

`read(scaled=True)` is the default. If the channel (else its group, else the file) has `NI_Number_Of_Scales` > 0 and
`NI_Scaling_Status` is not `scaled`, the NI scales are applied: **Linear** (`x * slope + intercept`) and
**Polynomial** (coefficients from `..._Polynomial_Coefficients_Size`, default 4), each reading either the raw data
(`..._Input_Source` = 0xFFFFFFFF, also when absent) or the output of an earlier scale; the last scale is the result.
The arithmetic is double precision whatever the data type, and the result is float64. Without scaling, `read`
returns the stored values with their stored dtype. `read(scaled=False)` is always the stored values.

Any other scale type (RTD, thermocouple, table, strain, ...) raises `UnsupportedScalingError` on a scaled read; the
raw read works.

### Truncated files

Writers append to the file while acquiring, and a crash or a copy can cut a file anywhere.

- A file that ends inside its last segment is read as far as it is complete: the complete values of every channel of
  the cut segment (for contiguous layout each channel keeps the complete values it has; for interleaved layout the
  complete rows), and nothing of a segment whose metadata is cut. `file.truncated` is `True`.
- Bytes at the end that are fewer than a lead-in (28 bytes) count as a cut as well.
- A last segment with next segment offset 0xFFFFFFFFFFFFFFFF (still being written) runs to the end of the file; it
  is truncated only if its data ends inside a chunk.
- `read(path, strict=True)` raises `TdmsFormatError` instead of any of these.
- Damage that is not at the end (a wrong signature in the middle, an index that does not fit its segment) is always
  a `TdmsFormatError`.


### Errors

All errors the package raises on purpose derive from `TdmsError`:

- `TdmsFormatError` (also a `ValueError`): not TDMS, or inconsistent: wrong signature, unknown version, a file shorter
  than a lead-in (including 0 bytes), counts and lengths that do not fit the file, an index of 0 without a previous
  index, a data type that changes between segments, interleaved channels with different counts, an invalid object
  path, a `.tdms_index` file.
- `UnsupportedFeatureError` (also a `NotImplementedError`): valid but not read: DAQmx raw data, extended floats.
  `UnsupportedScalingError` is one of these.

Problems of the file system (`FileNotFoundError`, `PermissionError`, ...) are not wrapped.

Fuzzed and damaged files may raise only these. They must not hang or allocate memory beyond what the file's size can
justify: every count and length in the file is checked against the bytes that remain before anything is allocated.

## Semantics of segments

These are the rules the implementation follows. Each has files among the test cases.

1. A file is a sequence of segments. A segment is a 28 byte lead-in (`TDSm`, table of contents, version, next segment
   offset, raw data offset), metadata, raw data. The table of contents is little-endian; the rest of the lead-in,
   the metadata and the data use the byte order its big-endian flag says, per segment.
2. The metadata is read only if the metadata flag is set. If the new object list flag is also set, the list replaces
   the active object list (the order of objects the raw data follows); without it, the objects listed only **update**
   what the previous list had (properties, raw data index) and the list stays as it was. Objects not listed keep their
   index and still get their share of the raw data.
3. A segment without metadata continues with the active list and indexes.
4. The raw data of a segment starts at lead-in + 28 + raw data offset and ends at next segment offset (or the end of
   the file when it is 0xFFFFFFFFFFFFFFFF).
5. A raw data index is `0xFFFFFFFF` (no data in this segment), `0` (the same type and count as the object's previous
   index), or explicit (length, type, dimension 1, number of values; strings add the total size in bytes). A channel's
   data type cannot change.
6. The raw data is a number of **chunks** (size of the data / size of one chunk), each holding every channel that has data,
   in list order: contiguous (channel after channel) or interleaved (the interleaved flag: one value of each channel in
   turn). Channels may have different counts when contiguous; interleaved needs equal counts. A string channel is its
   offsets (`uint32` end positions) then the text, UTF-8.
7. Object paths are `/`, `/'group'` or `/'group'/'channel'`. A group exists if any path mentions it.
8. Property updates: the last written value wins; properties are per object and accumulate over segments.

## How it is implemented

- `tdmspy/_metadata.py` parses a segment's metadata (objects, raw data indexes, properties). Every count and length is
  checked against the bytes that remain before anything is read or allocated.
- `tdmspy/_scan.py` walks the segments and keeps, per channel, a list of *blocks*: `repeat` units of `per`
  consecutive values, `stride` bytes apart, from byte `offset`. A segment of several chunks, or interleaved data, is
  one block; the identical segments of an acquisition merge into one (a file of 250 000 segments of one channel set
  is a handful of blocks). A segment whose metadata equals the previous segment's metadata is not parsed again.
- `tdmspy/file.py` reads a slice by mapping the file and copying the wanted part of the wanted blocks into a new
  array, so a slice costs what it returns, not what the channel holds. Byte order, booleans, complex numbers and
  timestamps are converted while copying.
- `tdmspy/_scaling.py` applies the NI scales.
- No global state; no I/O at import; no handle kept open.

## Development

- The catalogue (`tests/catalogue.py`) lists the special cases. A new case is added there, then as a file in
  `tests/testcases.py` with the values it must read as, and the file is stored in `tests/data/synthetic` (`just testcases`).
- `tests/builder.py` writes any byte sequence the format can express, including broken ones.
- Real files go into `tests/data/real`, listed in `REAL_FILES` with the cases they exercise; `just real-expected`
  stores what they contain.
- Test data must not identify what was measured, where, by whom or on which device: see `tests/test_privacy.py`.
