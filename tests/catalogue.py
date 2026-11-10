"""Every special case of the TDMS format that the reader has to go through.

This is the specification the test cases are measured against: `test_catalogue_coverage.py` fails if a case listed
here is not exercised by at least one synthetic case (`testcases.py`) or real file (`REAL_FILES` in
`testcases.py`), or if a case names an id that is not listed here. `doc/special-cases.md` is generated from it
(`just catalogue`). Add the case here first, then the file that exercises it.

An id is `<area>.<name>`; the value is what the reader must do.
"""

from __future__ import annotations

CATALOGUE: dict[str, str] = {
    # -- the file and its segments --------------------------------------------------------------
    "file.minimal": "A file of one segment with one group and one channel is read.",
    "file.empty_is_error": "A file of 0 bytes, or shorter than a lead-in (28 bytes), is a `TdmsFormatError`.",
    "file.wrong_signature_is_error": "A file that does not start with `TDSm` is a `TdmsFormatError` (not a crash, not a guess).",
    "file.index_file_is_error": "A `.tdms_index` file (signature `TDSh`) is refused with a clear message: it holds no data.",
    "file.not_tdms_inside": "A segment with a wrong signature after valid ones is a `TdmsFormatError`, unless it is the cut-off end of the file.",
    "version.4712": "Segments of version 4712 (TDMS 1.0) are read.",
    "version.4713": "Segments of version 4713 (TDMS 2.0) are read.",
    "version.unknown_is_error": "Any other version number is a `TdmsFormatError`.",
    # -- segments -------------------------------------------------------------------------------
    "segment.many": "Hundreds of small segments of one channel are read as one continuous channel, in order.",
    "segment.metadata_only": "A segment without raw data (no raw data flag) defines objects and properties and adds no values.",
    "segment.raw_only": "A segment without metadata (no metadata flag) holds more data for the previous segment's objects, layout and indexes.",
    "segment.meta_without_new_list": "A segment with metadata but without the new-object-list flag updates only the objects it lists; "
    "every other object keeps its index and still receives its share of the raw data.",
    "segment.new_list_replaces": "A new object list replaces the active one: an object left out gets no more data in that segment, one added gets some.",
    "segment.raw_offset_padding": "The raw data starts where the lead-in's raw data offset says, even if the metadata is shorter.",
    "segment.zero_values": "A channel whose index says 0 values (or a segment with an empty raw data section) adds nothing and breaks nothing.",
    "segment.next_offset_unknown": "A last segment whose next segment offset is 0xFFFFFFFFFFFFFFFF (still being written) runs to the end of the file.",
    "segment.truncated_file": "A file cut off inside its last segment gives the complete values of that segment and sets `file.truncated`; "
    "`strict=True` raises instead.",
    "segment.next_offset_beyond_end": "A last segment declaring more bytes than the file has is treated as truncated.",
    "segment.chunks": "A segment whose raw data is several repeats of the channel set (chunks) gives all the values, in order.",
    "segment.partial_chunk": "Bytes after the last whole chunk of a truncated segment give the values that are complete, none that are cut.",
    # -- the raw data index ---------------------------------------------------------------------
    "index.none": "An index of 0xFFFFFFFF means the object has no data in this segment (and is not an error).",
    "index.same_as_previous": "An index of 0 means the previous segment's type and count for that channel.",
    "index.changed_count": "A new explicit index changes a channel's count in a later segment.",
    "index.string": "The string index (type, dimension, count, total size) is read.",
    "index.type_change_is_error": "A channel whose data type changes between segments is a `TdmsFormatError`.",
    "index.same_without_previous_is_error": "An index of 0 for a channel that never had an explicit one is a `TdmsFormatError`.",
    # -- the layout of the raw data -------------------------------------------------------------
    "layout.contiguous_multi_channel": "Channels of one chunk are stored one after the other, each with its own count.",
    "layout.unequal_counts": "Channels in one segment may have different counts (contiguous layout).",
    "layout.interleaved": "Interleaved data (one value of each channel after the other) is read.",
    "layout.interleaved_chunks": "Interleaved data in several chunks is read.",
    "layout.interleaved_mixed_types": "Interleaved channels of different fixed sizes (int16 next to float64) are read.",
    "layout.interleaved_unequal_is_error": "Interleaved channels with different counts are a `TdmsFormatError`.",
    "layout.big_endian": "A segment with the big-endian flag is read completely: lead-in, metadata and data.",
    "layout.mixed_endianness": "Segments of different byte order in one file are each read in their own.",
    "layout.channel_added_later": "A channel that first appears in a later segment starts there.",
    "layout.channel_dropped": "A channel that stops getting data keeps what it has.",
    # -- data types -----------------------------------------------------------------------------
    "type.integers": "int8 to int64 and uint8 to uint64 channels are read exactly, as native numpy arrays.",
    "type.floats": "float32 and float64 channels are read exactly, including NaN, inf and -0.0.",
    "type.bool": "A boolean channel is read as a numpy bool array.",
    "type.string": "A string channel (offsets then text) is read, over several segments too, with empty and non-ASCII strings.",
    "type.timestamp": "A timestamp channel is read as `datetime64[us]` (UTC, truncated).",
    "type.complex": "Complex single and double channels are read as complex64 and complex128.",
    "type.no_data_type": "A channel that never has data has `dtype None`, length 0 and reads as an empty array (the quality_score channels).",
    "type.unsupported_is_error": "A data type this package cannot read (extended float) is an `UnsupportedFeatureError`, not garbage.",
    # -- properties -----------------------------------------------------------------------------
    "prop.all_types": "Properties of every type (integers, floats, bool, string, timestamp, complex, with-unit floats) are read as Python values.",
    "prop.root_group_channel": "Properties at the file, group and channel level are kept apart.",
    "prop.update": "A property written again in a later segment has the later value; others stay.",
    "prop.added_later": "A property that first appears in a later segment is there.",
    "prop.unicode": "Names and values with non-ASCII characters are read (UTF-8).",
    "prop.empty_and_odd_names": "An empty string value, a name with a trailing space and a very long value are kept as they are.",
    "prop.timestamp_exact": "A timestamp property keeps seconds and the 2**-64 fraction exactly; conversion to `datetime64` truncates.",
    "prop.many": "Thousands of properties on one object are read.",
    "prop.unsupported_type_is_error": "A property of a type this package cannot read is an `UnsupportedFeatureError`.",
    # -- objects and paths ----------------------------------------------------------------------
    "path.group_without_channels": "A group that only has properties is a group with no channels.",
    "path.implicit_group": "A channel whose group object is never written still has its group.",
    "path.order": "Groups and channels are in the order of their first appearance in the file, not sorted.",
    "path.quote_and_slash": "Names with `'` (written `''`), `/` and spaces are read back as written.",
    "path.unicode": "Group and channel names with non-ASCII characters are read.",
    "path.reserved_looking_names": "Groups or channels named `info`, `name`, `groups`, `properties`, `data` or `read` are ordinary.",
    "path.root_only_properties": "A file whose only object is the root has properties and no groups.",
    "path.invalid_is_error": "An object path that is not `/` or `/'group'` or `/'group'/'channel'` is a `TdmsFormatError`.",
    # -- scaling --------------------------------------------------------------------------------
    "scale.linear": "A channel with one linear NI scale reads scaled, and raw with `scaled=False`.",
    "scale.polynomial": "A polynomial NI scale is applied.",
    "scale.chain": "Scales that take another scale as their input source are applied in that order.",
    "scale.already_scaled": "`NI_Scaling_Status` = `scaled` means the data is final and is not scaled again.",
    "scale.group_level": "Scaling declared on the group applies to its channels without their own.",
    "scale.file_level": "Scaling declared on the file applies to channels and groups without their own.",
    "scale.channel_overrides_group": "A channel's own scaling wins over its group's.",
    "scale.float32_input": "Scaling single precision data is done in double precision and gives float64.",
    "scale.unsupported_is_error": "A scale type that is not implemented (RTD, thermocouple, ...) raises `UnsupportedScalingError` on a scaled read; "
    "the raw read works.",
    "scale.none": "A channel without scale properties reads the same scaled and raw.",
    # -- unsupported layouts --------------------------------------------------------------------
    "daqmx.is_error": "DAQmx raw data (indexes 0x1269 and 0x126A) is an `UnsupportedFeatureError` that names DAQmx.",
    # -- robustness -----------------------------------------------------------------------------
    "robust.huge_counts": "Declared counts and lengths far beyond the file's size are a `TdmsFormatError` without allocating them.",
    "robust.fuzz": "Truncated and randomly damaged files raise only `TdmsError` (never another exception, never hang, never allocate wildly).",
    "robust.lazy": "Opening a file reads its structure; channel data is read when asked for, and a slice reads only what it needs.",
    "robust.large": "A file of 100 MB opens quickly and a channel reads in time proportional to its size (slow test).",
    "robust.many_segments": "A file of tens of thousands of segments opens in a reasonable time.",
}


COVERED_BY_TESTS: dict[str, str] = {
    "robust.fuzz": "tests/test_fuzz.py::test_damaged_files_only_raise_tdms_errors",
    "robust.lazy": "tests/test_robust.py::test_opening_reads_the_structure_not_the_data",
    "robust.large": "tests/test_robust.py::test_a_hundred_megabyte_channel_reads_in_proportion",
}
"""Ids that are not a file but a behaviour, with the test that checks it (`test_catalogue_coverage.py` looks for it)."""
