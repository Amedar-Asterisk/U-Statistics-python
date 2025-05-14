import U_stats.tensor_contraction.path as ours
from U_stats.utils import einsum_expression_to_mode, Timer
import opt_einsum as oe
import einsum_benchmark as eb
import numpy as np

SIZE = 10**10


def get_shapes(expr: str):
    lhses, _ = einsum_expression_to_mode(expr)
    shapes = [(SIZE,) * len(lhs) for lhs in lhses]
    return shapes


def check(lhses, shapes):
    for lhs, shape in zip(lhses, shapes):
        if len(lhs) != len(shape):
            raise ValueError(f"Length of lhs {lhs} and shape {shape} do not match.")
        for dim in shape:
            if dim != SIZE:
                raise ValueError(f"Dimension {dim} is not equal to SIZE {SIZE}.")
    return True


def save_format(file_path: str):
    with open(file_path, "w") as f:
        for ins in eb.instances:
            f.write(f"{ins.name},{ins.format_string}\n")


def main(oe_method: str = "greedy"):
    for ins in eb.instances:
        computing_expr = ins.format_string
        lhses, _ = einsum_expression_to_mode(computing_expr)
        shapes = get_shapes(computing_expr)
        check(lhses, shapes)
        with Timer(name=f"oe opt for {ins.name}") as t:
            _, oes = oe.contract_path(
                computing_expr, *shapes, optimize=oe_method, shapes=True
            )

        with Timer(name=f"ours opt for {ins.name}") as t:
            _, oursinfo = ours.TensorExpression(mode=lhses).tupled_path(
                method="greedy", analyze=True
            )
        # print(
        #     f"oe opt cost: {oes.opt_cost:.2e}, largest intermediate: {oes.largest_intermediate:.2e}"
        # )
        # print(
        #     f"ours opt cost: {oursinfo[0]:.2e}, largest intermediate: {oursinfo[1]:.2e}"
        # )
        # print(f"compare cost: {oursinfo[0] / oes.opt_cost:.2e}")

        # print(
        #     f"compare largest intermediate: {oursinfo[1] / oes.largest_intermediate:.2e}"
        # )
        if oursinfo[1] > oes.largest_intermediate:
            # print(computing_expr)
            print(
                f"oe opt cost: {oes.opt_cost:.2e}, largest intermediate: {oes.largest_intermediate:.2e}"
            )
            print(
                f"ours opt cost: {oursinfo[0]:.2e}, largest intermediate: {oursinfo[1]:.2e}"
            )


if __name__ == "__main__":
    # a = "qøHa,Þõb,pqc,ąZd,Äe,ôf,g,h,RBwli,ÿBj,äk,þl,Sm,gAn,ÒhĈpo,ØÉþp,q,ìtr,gOs,t,u,KÔv,zkxw,ûx,ÀHgy,Íz,ÞCA,ÜB,C,ĆCùØD,êhèÊE,wF,G,H,eI,éàJ,IĂÌFiK,ÅÂDÝÙáôL,ÐæÂzM,N,hHO,ÐÊP,ÝkQ,ĈĀPR,BNíS,ûT,ZùbU,ÙÌýV,ÅGPóÕW,ÖX,XjOY,Z,ÏÀ,zÆoÁ,oÂ,EÝÃ,þCÄ,ÏÿÅ,ÔĈÆ,õÇ,ÕíĀùÝ×È,BÉ,yêBÂÊ,läË,ãtÔÌ,ãÍ,ÝaSÎ,ÄÏ,ÔÜpāXÐ,ýÑ,þÀÒ,WÜznÓ,úqÔ,ûnÕ,xÖ,MøêUÎ×,UØ,yßòpÙ,ÕðăIÚ,óÄöÛ,ÌÜ,Ý,Þ,ùß,Íà,ăqÌâçXóá,Utàâ,ã,Øä,ljÝå,Bæ,Búâç,åĀè,é,Jrê,üë,Ïøì,þí,ídî,éñï,Aoð,ÒĀñ,úò,RuÒó,ëÇÔÍøézô,õ,Îö,Iì÷,ø,Ïù,ú,ĆïÞû,ÚÍÁü,XĄÏý,þ,Zÿ,éĀ,sþúçcÎlā,hÖEĂ,Eîìsă,Ą,xHÜÂą,Ć,ëTßć,bĈ->"
    # lhses, _ = einsum_expression_to_mode(a)

    # paths, _ = ours.TensorExpression(mode=lhses).path(method="greedy")
    # for _, format_string in paths:
    #     print(format_string)
    #     lhses, rhs = einsum_expression_to_mode(format_string)
    #     print(len(rhs))
    main(oe_method="greedy")
