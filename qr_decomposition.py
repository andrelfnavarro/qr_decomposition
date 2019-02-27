from math import sqrt
import numpy as np
from copy import deepcopy


def imprime(A,n):
    '''Imprime a matriz A, recebida como parâmetro,
    com 'n' algarismos decimais - feito para melhorar
    visualização em casos que não necessitam de muita precisão
    após a vírgula.'''
    np.set_printoptions(precision=n)
    print(np.matrix(A))
    print("\n")

def matrixmult(A, B):
    '''Função responsável pela multiplicação das duas
    matrizes que recebe com parâmetro'''

    if len(A[0]) != len(B):
        print("Dimensoes invalidas.")
        return
    C = [[0 for i in range(len(B[0]))] for j in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):
                C[i][j] += A[i][k] * B[k][j]
    return C


def transposta(M):
    '''Função que recebe matriz M e retorna sua transposta.'''
    n = len(M)
    T = makeZero(len(M[0]), len(M))
    for i in range(n):
        for j in range(n):
            T[j][i] = M[i][j]
    return T

def modulo(v):
    '''Função que retorna o valor do módulo de um vetor. Soma
    todos seus elementos ao quadrado, depois retorna a raiz da soma.'''
    sum = 0
    for i in range(len(v)):
        sum += v[i] ** 2
    return sqrt(sum)



def moduloVector(v):
    '''Função que retorna o próprio vetor que recebe como
    parâmetro, porém normalizado. Utiliza a função modulo(v)
    para dividir cada elemento pelo módulo.'''
    if modulo(v) == 0:
        return v
    else:
        mod = 1 / modulo(v)
        for i in range(len(v)):
            v[i] *= mod
    return v


def moduloMat(A):
    '''Função que recebe uma matriz A como
    parâmetro e a retorna com suas colunas normalizadas
    (cada coluna tem seus elementos divididos pelo módulo
    da própria coluna).'''
    C = transposta(A)
    for i in range(len(C)):
        for j in range(len(C[i])):
            C[i] = moduloVector(C[i])
    D = transposta(C)
    return D


def makeZero(m, n):
    '''Função que recebe duas ordens e retorna uma matriz
    nula de ordem m x n.'''
    C = [[0 for lin in range(n)] for col in range(m)]
    return C


def makeNullVector(n):
    '''Função que recebe tamanho n e retorna
    vetor de zeros, com len == n'''
    Q = [[0.0] * n for i in range(n)]
    return Q


def makeId(m):
    '''Função que recebe ordem de matriz m e retorna
    uma matriz identidade de ordem m'''
    C = [[0 for lin in range(m)] for col in range(m)]
    for i in range(m):
        for j in range(m):
            C[i][j] = float(i == j)
    return C


def signal(a, b):
    return (a > b) - (a < b)

def HH(R, column):
    '''Funcao recebe uma matriz, a coluna em que será realizada as operações
    e realiza o método de Householder. Além disso, ela realiza o recorte da matriz
    para as operações e retorna o Qi (Q da iteração).'''
    n = len(R)
    I = makeId(n)
    # criamos o vetor 'a', o vetor 'e' e o escalar delta
    a = transposta(R)[column][column:]
    e = transposta(I)[column][column:]
    delta = -signal(a[0], 0)

    # criamos aqui o vetor v = a + delta * modulo(a) * e
    v = []
    for i in range(n - column):
        v.append(a[i] + delta * modulo(a) * e[i])
    v = moduloVector(v)

    # Faz a matriz Q_i recortada para mnimizar as operacoes (dentro do tamanho de matriz certo para a iteração)
    Q_i_cortada = makeZero(n,n)
    for i in range(n - column):
        for j in range(n - column):
            Q_i_cortada[i][j] = I[i][j] - 2 * v[i] * v[j]


    #faz uma matriz Q_i, do tamanho n x n e retorna valores ou da identidade ou valores da própria matriz recortada
    #anteriormente
    Q_i = makeZero(n, n)
    for i in range(n):
        for j in range(n):
            if i < column or j < column:
                Q_i[i][j] = float(i == j)
            else:
                Q_i[i][j] = Q_i_cortada[i - column][j - column]

    return Q_i

def QR(A):
    """Funcao responsavel por calcular, atraves do metodo QR
    sugerido no pdf do EP, as matrizes Q e R finais, a partir
    dos Qis fornecidos pela funcao HH (householder)."""
    n = len(A)
    R = deepcopy(A)
    Q = makeId(n)

    for column in range(n - 1):
        Q_i = HH(R, column)
        # Hv1 * ... * Hvn-2 * Hvn-1 =Q
        Q = matrixmult(Q_i, Q)
        # Hvn−1 *...* Hv2 * Hv1 * A = R
        R = matrixmult(Q_i, R)

    # Q = (Hvn−1*...*Hv2*Hv1)T = Hv1*Hv2*...*Hvn−1.
    # Logo, como calculamos até aqui (Hvn−1*...*Hv2*Hv1)T, deveremos retornar sua tranposta.
    return transposta(Q), R

def eigvalues(A, n):
    '''Função que recebe matriz A e retorna a matriz de autovalores,
    através da utilização da função QR 'n' vezes.'''
    X = deepcopy(A)
    # cria cópia de A para usar no funcionamento
    # Para a iteração, recebe Q e R da matriz X e logo a atualiza por meio
    #do explicado no pdf:
    # A_n = R_n-1*Q_n-1, sendo essa aproximada da matriz de autovalores ao longo das iterações
    for i in range(n):
        Q, R = QR(X)
        X = matrixmult(R, Q)
    return X


def scalVector(u, v):
    '''Retorna o produto escalar de
    dois vetores: u e v.'''
    soma = 0
    for i in range(len(u)):
        soma += u[i] + v[i]
    return soma


def eigvector(A, n):
    '''Função que recebe matriz A e retorna a matriz de autovetores,
    através da utilização da função QR 'n' vezes.'''
    X = deepcopy(A)
    #cria cópia de A para usar no funcionamento
    V = makeId(len(A))
    #cria matriz identidade V, a qual em cada iteração é multiplicada
    #pelo Q da nova iteração. Além disso, atualiza a matriz principal
    #assim como era feito no pdf:
    # A_n = R_n-1*Q_n-1

    #Além disso, a matriz V, de autovetores, é dada pela multiplicação de todos
    #os Q's obtidos: Q_1 * Q_2 *...* Q_n
    for i in range(n):
        Q, R = QR(X)
        X = matrixmult(R, Q)
        V = matrixmult(V, Q)
    return moduloMat(V)

def main():
    #PARA AUMENTAR NUMERO DE CASAS DECIMAIS, BASTA AUMENTAR O NUMERO PARAMETRO DA FUNCAO 'IMPRIME'
    print("EXERCÍCIO 1 - todos valores foram obtidos para n = numero de iterações = 20\nitem a)")
    A_1a = [[-2, 1, 0, 0, 0], [1, -2, 1, 0, 0], [0, 1, -2, 1, 0],[0, 0, 1, -2, 1],[0, 0, 0, 1, -2]]
    print("Matriz A:")
    imprime(A_1a, 1)
    print("Matriz de autovalores com duas casas decimais:")
    imprime(eigvalues(A_1a, 20), 2)
    print("Matriz de autovetores com duas casas decimais:")
    imprime(eigvector(A_1a, 20), 2)

    print("item d)")
    A_1a = [[-2,1,0,0,0,0,0], [1,-11,10,0,0,0,0], [0,10,-11,1,0,0,0], [0,0,1,-2,1,0,0], [0,0,0,1,-2,1,0],[0,0,0,0,1,-2,1],[0,0,0,0,0,1,-2]]
    print("Matriz A:")
    imprime(A_1a, 1)
    print("Matriz de autovalores com duas casas decimais:")
    imprime(eigvalues(A_1a, 20), 2)
    print("Matriz de autovetores com duas casas decimais:")
    imprime(eigvector(A_1a, 20), 2)

    print("EXERCÍCIO 2\nitem a)")
    A_1a = [[3,4,0], [4,3,0], [0,0,2]]
    print("Matriz A:")
    imprime(A_1a, 1)
    print("Matriz de autovalores com cinco casas decimais:")
    imprime(eigvalues(A_1a, 20), 6)
    print("Matriz de autovetores com seis casas decimais:")
    imprime(eigvector(A_1a, 20), 6)

    print("item b)")
    A_1a = [[3, 4, 0], [1, 3, 0], [0, 0, 2]]
    print("Matriz A:")
    imprime(A_1a, 1)
    print("Matriz de autovalores com seis casas decimais:")
    imprime(eigvalues(A_1a, 20), 6)

    print("item c)")
    A_1a = [[1, 1], [-3, 1]]
    print("Matriz A:")
    imprime(A_1a, 1)
    print("Matriz de autovalores com seis casas decimais:")
    imprime(eigvalues(A_1a, 20), 6)

    print("item d)")
    A_1a = [[3, 3], [0.33333, 5]]
    print("Matriz A:")
    imprime(A_1a, 5)
    print("Matriz de autovalores com seis casas decimais:")
    imprime(eigvalues(A_1a, 20), 6)


if __name__ == '__main__':
    main()