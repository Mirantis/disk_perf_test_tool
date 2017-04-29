#include <algorithm>
#include <cstdint>
#include <cstdio>

extern "C"
void interpolate_ts_on_seconds_border(
    unsigned int input_size,
    unsigned int output_size,
    uint64_t * times,
    uint64_t * values,
    unsigned int time_scale_coef,
    uint64_t * output)
{
    auto first_output_cell = (*times / time_scale_coef) * time_scale_coef;
    auto input_cell_end = *times - time_scale_coef;  // hack to unify loop
    uint64_t input_cell_begin;

//    std::printf("first_output_cell = %ld\n", (long)first_output_cell);

    for(auto curr_times=times, curr_data_ptr=values;
        curr_times < times + input_size ; ++curr_times, ++curr_data_ptr)
    {
        // take next cell from input array and calculate data rate in it
        auto data_left = *curr_data_ptr;
        input_cell_begin = input_cell_end;
        input_cell_end = *curr_times;

        auto rate = data_left / double(input_cell_end - input_cell_begin);

//        std::printf("input_cell_begin=%ld input_cell_end=%ld\n", (long)input_cell_begin, (long)input_cell_end);
//        std::printf("rate = %lf data_left=%ld\n", rate, (long)data_left);

        uint32_t first_output_cell_idx;
        if (input_cell_begin <= first_output_cell)
            first_output_cell_idx = 0;
        else
            // +1 because first_output_cell is actually the end of first cell
            first_output_cell_idx = (input_cell_begin - first_output_cell) / time_scale_coef + 1;

        uint32_t last_output_cell_idx = (input_cell_end - first_output_cell) / time_scale_coef;

        if ((input_cell_end - first_output_cell) % time_scale_coef != 0)
            ++last_output_cell_idx;

        last_output_cell_idx = std::min(last_output_cell_idx, output_size - 1);

//        std::printf("fidx=%d lidx=%d\n", (int)first_output_cell_idx, (int)last_output_cell_idx);

        for(auto output_idx = first_output_cell_idx; output_idx <= last_output_cell_idx ; ++output_idx)
        {
            // current output cell time slot
            auto out_cell_begin = output_idx * time_scale_coef + first_output_cell - time_scale_coef;
            auto out_cell_end = out_cell_begin + time_scale_coef;
            auto slot = std::min(out_cell_end, input_cell_end) - std::max(out_cell_begin, input_cell_begin);

            auto slice = uint64_t(rate * slot);

//            std::printf("slot=%ld slice=%lf output_idx=%ld\n", (long)slot, (double)slice, (long)output_idx);

            data_left -= slice;
            output[output_idx] += slice;
        }
        output[last_output_cell_idx] += data_left;
    }
}


extern "C"
void interpolate_ts_on_seconds_border_v2(
    unsigned int input_size,
    unsigned int output_size,
    uint64_t * times,
    uint64_t * values,
    unsigned int time_step,
    uint64_t * output)
{
    auto output_end = (*times / time_step) * time_step;
    auto output_begin = output_end - time_step;
    auto output_cell = output;

    auto input_cell = values;
    auto input_time = times;
    auto input_val = *input_cell;
    auto input_begin = *input_time - time_step;
    auto input_end = *input_time;
    auto rate = ((double)*input_cell) / (input_end - input_begin);

    // output array mush fully cover input array
    while(output_cell < output + output_size) {
        // check if cells intersect
        auto intersection = ((int64_t)std::min(output_end, input_end)) - std::max(output_begin, input_begin);

        // add intersection slice to output array
        if(intersection > 0) {
            auto slice = (uint64_t)(intersection * rate);
            *output_cell += slice;
            input_val -= slice;
        }

        // switch to next input or output cell
        if (output_end >= input_end){
            *output_cell += input_val;

            ++input_cell;
            ++input_time;

            if(input_time == times + input_size)
                return;

            input_val = *input_cell;
            input_begin = input_end;
            input_end = *input_time;
            rate = ((double)*input_cell) / (input_end - input_begin);
        } else {
            ++output_cell;
            output_begin = output_end;
            output_end += time_step;
        }
    }
}