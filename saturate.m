function out = saturate(in,high,low)

out = min(high, max(in, low));
end