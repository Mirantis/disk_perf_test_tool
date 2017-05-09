#include <algorithm>
#include <cstdint>
//#include <cstdio>


uint64_t round(double x) {
    return (uint64_t)(x + 0.5);
}


extern "C"
unsigned int interpolate_ts_on_seconds_border(unsigned int input_size,
                                              unsigned int output_size,
                                              const uint64_t * times,
                                              const uint64_t * values,
                                              unsigned int time_step,
                                              uint64_t * output)
{
    auto output_begin = *times - time_step;
    auto output_end = *times;

    auto input_begin = *times - time_step;
    auto input_end = *times;

    auto output_cell = output;

    auto input_cell = values;
    auto input_val = *input_cell;

    auto input_time = times;
    auto rate = ((double)*input_cell) / (input_end - input_begin);

    // output array mush fully cover input array
    while(output_cell < output + output_size) {
        // check if cells intersect
        auto intersection = ((int64_t)std::min(output_end, input_end)) - std::max(output_begin, input_begin);

        // add intersection slice to output array
        if(intersection > 0) {
            auto slice = std::min(input_val, round(intersection * rate));
            *output_cell += slice;
            input_val -= slice;
        }

        // switch to next input or output cell
        if (output_end >= input_end){
            *output_cell += input_val;

            ++input_cell;
            ++input_time;

            if(input_time == times + input_size)
                return output_cell - output + 1;

            if (output_end == input_end) {
                ++output_cell;
                output_begin = output_end;
                output_end += time_step;
            }

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
    return output_size;
}


extern "C" unsigned int interpolate_ts_on_seconds_border_qd(unsigned int input_size,
                                                            unsigned int output_size,
                                                            const uint64_t * times,
                                                            const uint64_t * values,
                                                            unsigned int time_step,
                                                            uint64_t * output)
{
    auto input_end = times + input_size;
    auto curr_output_tm = *times - time_step / 2;

    for(auto output_cell = output; output_cell < output + output_size; ++output_cell) {
        while (curr_output_tm > *times) {
            if (++times >= input_end)
                return output_cell - output;
            ++values;
        }
        *output_cell = *values;
        curr_output_tm += time_step;
    }

    return output_size;
}


extern "C" int interpolate_ts_on_seconds_border_fio(unsigned int input_size,
                                                    unsigned int output_size,
                                                    const uint64_t * times,
                                                    unsigned int time_step,
                                                    uint64_t * output_idx,
                                                    uint64_t empty_cell_placeholder,
                                                    bool allow_broken_step)
{
    auto input_end = times + input_size;
    auto output_end = output_idx + output_size;

    float no_step = time_step * (allow_broken_step ? 0.3 : 0.1);
    float more_then_step = time_step * 1.9;
    float step_min = time_step * 0.9;
    float step_max = time_step * (allow_broken_step ? 1.9 : 1.1);

    auto curr_input_tm = times;
    long int curr_output_tm = *curr_input_tm - time_step;

    for(; output_idx < output_end; ++output_idx) {

        // skip repetition of same time
        while(((long int)*curr_input_tm - curr_output_tm) <= no_step and curr_input_tm < input_end)
            ++curr_input_tm;

        if (curr_input_tm == input_end)
            break;

        long int dt = *curr_input_tm - curr_output_tm;
//        std::printf("dt=%ld curr_input_tm=%lu curr_output_tm=%ld\n", dt, *curr_input_tm, curr_output_tm);

        if (dt <= step_max and (dt > step_min or allow_broken_step)) {
            *output_idx = curr_input_tm - times;
        } else if (dt >= more_then_step or (allow_broken_step and dt >= step_max)) {
            *output_idx = empty_cell_placeholder;
        } else
            return -(int)(curr_input_tm - times);

        curr_output_tm += time_step;
    }

    return output_size - (output_end - output_idx);
}