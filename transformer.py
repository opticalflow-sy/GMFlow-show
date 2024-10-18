class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(
            self,
            d_model=256,
            nhead=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            with_shift=False,
            **kwargs,
    ):
        super(TransformerBlock, self).__init__()

        # 自注意力层
        self.self_attn = TransformerLayer(
            d_model=d_model,
            nhead=nhead,
            attention_type=attention_type,
            no_ffn=True,
            ffn_dim_expansion=ffn_dim_expansion,
            with_shift=with_shift,
        )
        # 交叉注意力层+FFN层
        self.cross_attn_ffn = TransformerLayer(
            d_model=d_model,
            nhead=nhead,
            attention_type=attention_type,
            ffn_dim_expansion=ffn_dim_expansion,
            with_shift=with_shift,
        )

    def forward(
            self,
            source,
            target,
            height=None,
            width=None,
            shifted_window_attn_mask=None,
            attn_num_splits=None,
            **kwargs,
    ):
        # source, target: [B, L, C]

        # self attention
        source = self.self_attn(
            source,
            source,
            height=height,
            width=width,
            shifted_window_attn_mask=shifted_window_attn_mask,
            attn_num_splits=attn_num_splits,
        )

        # cross attention and ffn
        source = self.cross_attn_ffn(
            source,
            target,
            height=height,
            width=width,
            shifted_window_attn_mask=shifted_window_attn_mask,
            attn_num_splits=attn_num_splits,
        )

        return source


class FeatureTransformer(nn.Module):
    def __init__(
            self,
            num_layers=6,
            d_model=128,
            nhead=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            **kwargs,
    ):
        super(FeatureTransformer, self).__init__()

        self.attention_type = attention_type

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    attention_type=attention_type,
                    ffn_dim_expansion=ffn_dim_expansion,
                    with_shift=True
                    if attention_type == "swin" and i % 2 == 1
                    else False,
                )
                for i in range(num_layers)
            ]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            feature0,
            feature1,
            attn_num_splits=None,
            **kwargs,
    ):
        b, c, h, w = feature0.shape
        assert self.d_model == c

        feature0 = feature0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        feature1 = feature1.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]

        if self.attention_type == "swin" and attn_num_splits > 1:
            # global and refine use different number of splits
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=feature0.device,
            )  # [K*K, H/K*W/K, H/K*W/K]
        else:
            shifted_window_attn_mask = None

        # concat feature0 and feature1 in batch dimension to compute in parallel
        concat0 = torch.cat((feature0, feature1), dim=0)  # [2B, H*W, C]
        concat1 = torch.cat((feature1, feature0), dim=0)  # [2B, H*W, C]

        for layer in self.layers:
            concat0 = layer(
                concat0,
                concat1,
                height=h,
                width=w,
                shifted_window_attn_mask=shifted_window_attn_mask,
                attn_num_splits=attn_num_splits,
            )

            # update feature1
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

        feature0, feature1 = concat0.chunk(chunks=2, dim=0)  # [B, H*W, C]

        # reshape back
        feature0 = (
            feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        )  # [B, C, H, W]
        feature1 = (
            feature1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        )  # [B, C, H, W]

        return feature0, feature1