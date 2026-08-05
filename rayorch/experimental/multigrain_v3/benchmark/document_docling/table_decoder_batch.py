"""Active-mask batched decoding for Docling's unmodified TableFormer core."""

from __future__ import annotations

from typing import Any


def predict_tableformer_batch(model: Any, images: Any) -> list[tuple[Any, Any, Any]]:
    """Reproduce ``TableModel04_rs.predict`` for a batch of table images.

    The model weights and core modules are reused unchanged.  Only the Python
    autoregressive control state is vectorized; finished samples keep emitting
    ``<end>`` until the longest sample finishes, and each sequence is trimmed at
    its first end token before bbox decoding.
    """

    import torch

    model._tag_transformer.eval()
    enc_out = model._encoder(images)
    word_map = model._init_data["word_map"]["word_map_tag"]
    n_heads = model._tag_transformer._n_heads
    encoder_out = model._tag_transformer._input_filter(
        enc_out.permute(0, 3, 1, 2)
    ).permute(0, 2, 3, 1)
    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(-1)
    enc_inputs = encoder_out.view(batch_size, -1, encoder_dim).to(model._device)
    enc_inputs = enc_inputs.permute(1, 0, 2)
    positions = enc_inputs.shape[0]
    encoder_mask = torch.zeros(
        (batch_size * n_heads, positions, positions),
        dtype=torch.bool,
        device=model._device,
    )
    transformer_memory = model._tag_transformer._encoder(
        enc_inputs,
        mask=encoder_mask,
    )

    start_tag = word_map["<start>"]
    end_tag = word_map["<end>"]
    decoded_tags = torch.full(
        (1, batch_size),
        start_tag,
        dtype=torch.long,
        device=model._device,
    )
    cache = None
    finished = [False] * batch_size
    end_offsets: list[int | None] = [None] * batch_size
    skip_next_tag = [True] * batch_size
    prev_tag_ucel = [False] * batch_size
    first_lcel = [True] * batch_size
    bboxes_to_merge: list[dict[int, int]] = [dict() for _ in range(batch_size)]
    current_bbox = [-1] * batch_size
    bbox_index = [0] * batch_size
    tag_h_buffers: list[list[Any]] = [[] for _ in range(batch_size)]

    bbox_tags = {
        word_map["fcel"],
        word_map["ecel"],
        word_map["ched"],
        word_map["rhed"],
        word_map["srow"],
        word_map["nl"],
        word_map["ucel"],
    }
    skip_tags = {word_map["nl"], word_map["ucel"], word_map["xcel"]}

    for _ in range(model._max_pred_len):
        decoded_embedding = model._tag_transformer._embedding(decoded_tags)
        decoded_embedding = model._tag_transformer._positional_encoding(
            decoded_embedding
        )
        decoded, cache = model._tag_transformer._decoder(
            decoded_embedding,
            transformer_memory,
            cache,
            memory_key_padding_mask=encoder_mask,
        )
        logits = model._tag_transformer._fc(decoded[-1, :, :])
        next_tags = logits.argmax(1).tolist()

        for index, tag in enumerate(next_tags):
            if finished[index]:
                next_tags[index] = end_tag
                continue
            # The upstream implementation never increments line_num, so its
            # first-line xcel correction applies throughout decoding.
            if tag == word_map["xcel"]:
                tag = word_map["lcel"]
            if prev_tag_ucel[index] and tag == word_map["lcel"]:
                tag = word_map["fcel"]
            next_tags[index] = tag
            if tag == end_tag:
                finished[index] = True
                end_offsets[index] = decoded_tags.size(0)
                continue

            hidden = decoded[-1, index : index + 1, :]
            if not skip_next_tag[index] and tag in bbox_tags:
                tag_h_buffers[index].append(hidden)
                if not first_lcel[index]:
                    bboxes_to_merge[index][current_bbox[index]] = bbox_index[index]
                bbox_index[index] += 1

            if tag != word_map["lcel"]:
                first_lcel[index] = True
            else:
                if first_lcel[index]:
                    tag_h_buffers[index].append(hidden)
                    first_lcel[index] = False
                    current_bbox[index] = bbox_index[index]
                    bboxes_to_merge[index][current_bbox[index]] = -1
                    bbox_index[index] += 1

            skip_next_tag[index] = tag in skip_tags
            prev_tag_ucel[index] = tag == word_map["ucel"]

        decoded_tags = torch.cat(
            [
                decoded_tags,
                torch.tensor(
                    next_tags,
                    dtype=torch.long,
                    device=model._device,
                ).unsqueeze(0),
            ],
            dim=0,
        )
        if all(finished):
            break

    results = []
    for index in range(batch_size):
        end_offset = end_offsets[index]
        stop = decoded_tags.size(0) if end_offset is None else end_offset + 1
        sequence = decoded_tags[:stop, index].tolist()
        if model._bbox:
            outputs_class, outputs_coord = model._bbox_decoder.inference(
                enc_out[index : index + 1],
                tag_h_buffers[index],
            )
            outputs_class, outputs_coord = _merge_span_bboxes(
                model,
                outputs_class,
                outputs_coord,
                bboxes_to_merge[index],
            )
        else:
            outputs_class, outputs_coord = None, None
        results.append((sequence, outputs_class, outputs_coord))
    return results


def _merge_span_bboxes(
    model: Any,
    outputs_class: Any,
    outputs_coord: Any,
    bboxes_to_merge: dict[int, int],
) -> tuple[Any, Any]:
    """Reproduce TableModel04_rs horizontal-span bbox merging per sample."""

    import torch

    merged_classes = []
    merged_coords = []
    boxes_to_skip = []
    for box_index in range(len(outputs_coord)):
        first_box = outputs_coord[box_index].to(model._device)
        first_class = outputs_class[box_index].to(model._device)
        if box_index in bboxes_to_merge:
            second_index = bboxes_to_merge[box_index]
            second_box = outputs_coord[second_index].to(model._device)
            boxes_to_skip.append(second_index)
            merged_coords.append(
                model.mergebboxes(first_box, second_box).to(model._device)
            )
            merged_classes.append(first_class)
        elif box_index not in boxes_to_skip:
            merged_coords.append(first_box)
            merged_classes.append(first_class)
    classes = (
        torch.stack(merged_classes)
        if merged_classes
        else torch.empty(0)
    )
    coords = (
        torch.stack(merged_coords)
        if merged_coords
        else torch.empty(0)
    )
    return classes, coords
