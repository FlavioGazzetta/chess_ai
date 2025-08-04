# viewer.py ─ show every move, one board, no flicker
from __future__ import annotations
import json, queue, sys, time, pygame, chess, threading
from colorama import Fore, Style, init as _cinit
_cinit(autoreset=True)

# ───── constants ──────────────────────────────────────────────────────────
SQ = 72; BOARD_PX = 8*SQ
LIGHT, DARK = (240,217,181), (181,136, 99)
FRAME = (70,70,70); FPS = 60
UNICODE = {"P":"♙","N":"♘","B":"♗","R":"♖","Q":"♕","K":"♔",
           "p":"♟","n":"♞","b":"♝","r":"♜","q":"♛","k":"♚"}
QUEUE: "queue.Queue[str]|None" = None

# ───── helpers ────────────────────────────────────────────────────────────
_font_cache:dict[int,pygame.font.Font]={}
def _font(sz:int)->pygame.font.Font:
    if sz not in _font_cache:
        _font_cache[sz] = pygame.font.SysFont("DejaVuSans", sz, bold=True)
    return _font_cache[sz]

def _background()->pygame.Surface:
    surf=pygame.Surface((BOARD_PX,BOARD_PX))
    surf.fill(FRAME)
    for r in range(8):
        for c in range(8):
            col = LIGHT if (r+c)&1 else DARK
            pygame.draw.rect(surf,col,(c*SQ,r*SQ,SQ,SQ))
    return surf

def _piece_imgs()->dict[str,pygame.Surface]:
    imgs={}
    for sym,uni in UNICODE.items():
        col=(255,255,255) if sym.isupper() else (25,25,25)
        txt=_font(int(SQ*0.8)).render(uni,True,col)
        s=pygame.Surface((SQ,SQ),pygame.SRCALPHA); s.blit(txt,txt.get_rect(center=(SQ//2,SQ//2)))
        imgs[sym]=s
    return imgs

# ───── viewer main loop ───────────────────────────────────────────────────
def view_moves(shared:queue.Queue[str])->None:
    global QUEUE; QUEUE=shared
    pygame.init(); win=pygame.display.set_mode((BOARD_PX,BOARD_PX))
    pygame.display.set_caption("Chess Viewer"); clock=pygame.time.Clock()

    back   = _background();  piece_img=_piece_imgs()
    board  : chess.Board|None=None
    last   : dict[int,chess.Piece]={}

    running=True
    while running:
        # pull **one** message (blocking ≤1/FPS s) so we repaint each ply
        try: raw=QUEUE.get(timeout=1/FPS)
        except queue.Empty: raw=None

        if raw:
            try: fen=json.loads(raw)["fen"]
            except Exception as e:
                print(Fore.RED+f"[viewer] bad msg: {e}"+Style.RESET_ALL); continue

            # start/replace board when new game or first FEN
            if board is None or board.is_game_over():
                board=chess.Board(fen); last={}
                back=_background()      # fresh clean board
            else:
                board.set_fen(fen)

            new_map=board.piece_map()
            changed={sq for sq in set(last)|set(new_map) if last.get(sq)!=new_map.get(sq)}
            last=new_map

            for sq in changed:
                r,c=divmod(sq,8)
                col = LIGHT if (r+c)&1 else DARK
                pygame.draw.rect(back,col,(c*SQ,r*SQ,SQ,SQ))
                if piece:=new_map.get(sq):
                    back.blit(piece_img[piece.symbol()],(c*SQ,r*SQ))

            win.blit(back,(0,0))
            pygame.display.flip()         # immediate render after this move

        # handle quit
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: running=False
        clock.tick(FPS)

    pygame.quit(); sys.exit()

# standalone quick-test
if __name__=="__main__":
    import multiprocessing as mp, random
    q=mp.Manager().Queue()
    threading.Thread(target=view_moves,args=(q,),daemon=True).start()
    b=chess.Board(); q.put(json.dumps({"fen":b.fen()}))
    while not b.is_game_over():
        b.push(random.choice(list(b.legal_moves)))
        q.put(json.dumps({"fen":b.fen()})); time.sleep(0.3)
